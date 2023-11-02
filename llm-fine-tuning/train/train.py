#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # new add
    # lr_scheduler_type: str = field(default='linear')
    # num_cycles: int = field(default=0.5, metadata={"help": "Number of cycles"})

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     from torch.utils.data import DataLoader, RandomSampler
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=128,  # 替换为你实际需要的batch size
#         sampler=RandomSampler(train_dataset)  # 在每个epoch开始前shuffle数据
#     )
#     return dict(train_dataset=train_dataset, train_dataloader=train_dataloader, eval_dataset=None, data_collator=data_collator)


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, test_size=0.1) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     from sklearn.model_selection import train_test_split

#     full_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
#     train_dataset, eval_dataset = train_test_split(full_dataset, test_size=0.2,random_state=42)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)




def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
   
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf", # model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "tatsu-lab/alpaca-7b-wdiff", # model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    # ########################################################################################################################
    # from transformers import get_scheduler, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup
    # from transformers import Trainer as HfTrainer 
    # import math
    # from typing import Callable, Iterable, Optional, Tuple, Union
    # import torch
    # from torch.optim import Optimizer
    # from torch.optim.lr_scheduler import LambdaLR
    # from transformers.trainer_utils import SchedulerType
    # from transformers.utils import logging

    # logger = logging.get_logger(__name__)
    
    # TYPE_TO_SCHEDULER_FUNCTION = {
    #     SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    #     SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    #     SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    #     SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    #     SchedulerType.CONSTANT: get_constant_schedule,
    #     SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    # }
    
    # def get_scheduler(
    #     name: Union[str, SchedulerType],
    #     optimizer: Optimizer,
    #     num_warmup_steps: Optional[int] = None,
    #     num_training_steps: Optional[int] = None,
    #     num_cycles: Optional[int] = None,
    # ):
    #     name = SchedulerType(name)
    #     schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    #     if name == SchedulerType.CONSTANT:
    #         return schedule_func(optimizer)
    
    #     # All other schedulers require `num_warmup_steps`
    #     if num_warmup_steps is None:
    #         raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
    
    #     if name == SchedulerType.CONSTANT_WITH_WARMUP:
    #         return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    #     # All other schedulers require `num_training_steps`
    #     if num_training_steps is None:
    #         raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    
    #     return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    # # rewrite the trainer
    # class Trainer(HfTrainer):
    #     def create_optimizer_and_scheduler(self, num_training_steps: int):
    #         if not hasattr(self.optimizer, 'param_groups'):
    #             super().create_optimizer_and_scheduler(num_training_steps)
                
    #         scheduler_name = self.args.lr_scheduler_type
    #         if scheduler_name:
    #             self.lr_scheduler = get_scheduler(
    #                 name=scheduler_name,
    #                 optimizer=self.optimizer,
    #                 num_warmup_steps=self.args.warmup_steps,
    #                 num_training_steps=num_training_steps,
    #                 num_cycles=self.args.num_cycles,  # This needs to be set in TrainingArguments
    #             )
    #         else:
    #             None
                
    # class Trainer(HfTrainer):
    #     def create_optimizer_and_scheduler(self, num_training_steps: int):
    #         if not hasattr(self.optimizer, 'param_groups'):
    #             super().create_optimizer_and_scheduler(num_training_steps)
                
    #             scheduler_name = self.args.lr_scheduler_type
    #             if scheduler_name:
    #                 self.lr_scheduler = get_scheduler(
    #                     name=scheduler_name,
    #                     optimizer=self.optimizer,
    #                     num_warmup_steps=self.args.warmup_steps,
    #                     num_training_steps=num_training_steps,
    #                     num_cycles=self.args.num_cycles,  # This needs to be set in TrainingArguments
    #                 )


    ########################################################################################################################
    # from transformers import get_scheduler
    # from transformers.optimization import AdamW
    # class LRBenchTrainer(Trainer):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.last_loss = []
    
    #     def training_step(self, model, inputs):
    #         # 使用父类的training_step方法计算loss
    #         output = super().training_step(model, inputs)
    #         # 保存最近的loss值
    #         self.last_loss.append(output.loss)

    #     def create_optimizer_and_scheduler(self, num_training_steps: int):
    #         self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
    
    #         self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps
    #         )

    #     ############################################
    #     def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, 
    #                                                              num_training_steps: int, last_epoch: int = -1):
    
    #         def lr_lambda(current_step):
    #             if current_step < num_warmup_steps:
    #                 return float(current_step) / float(max(1, num_warmup_steps))
                
    #             progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    #             if progress >= 1.0:
    #                 return 0.0
    #             return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
        
    #         return LambdaLR(optimizer, lr_lambda, last_epoch)
    ########################################################################################################################
    # from transformers import get_scheduler, get_cosine_schedule_with_warmup
    # from transformers.optimization import AdamW
    # from torch import nn
    # from typing import Callable, Iterable, Optional, Tuple, Union
    # from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
    # class CTrainer(Trainer):
    #     def create_optimizer_and_scheduler(self, num_training_steps: int):
    #         self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
    
    #         # self.lr_scheduler = get_scheduler(
    #         #     "cosine",  
    #         #     self.optimizer,
    #         #     num_warmup_steps=self.args.warmup_steps,
    #         #     num_training_steps=num_training_steps
    #         # )
    #         self.lr_scheduler = get_cosine_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps,
    #             num_cycles = 3.5
    #         )
    #     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
    #         # 调用原始的training_step
    #         step_output = super().training_step(model, inputs)
            
    #         # 记录当前的学习率
    #         logging.info("Current learning rate: %s", self.lr_scheduler.get_last_lr()[0])
    #         return step_output 


    
    
    from transformers import Trainer as HfTrainer
    from torch.utils.data.dataloader import DataLoader
    import logging
    import warnings
    # 配置 logging
    logging.basicConfig(filename='/a/bear.cs.fiu.edu./disk/bear-a/users/hjin008/Desktop/lrbench4llm_w_results/lrbench4llm/experiment_and_result/train_shuffle_log.txt', level=logging.INFO)
    
    class MyTrainer(HfTrainer):
        def on_epoch_begin(self, epoch, **kwargs):
            super().on_epoch_begin(epoch, **kwargs)

            # 获取前 5 个数据样本
            for i, batch in enumerate(self.train_dataloader):
                if i >= 5:
                    break
                print(batch) 
                warnings.warn(batch)
                raise Exception(batch)
                deepspeed.logging.info(batch)
                deepspeed.logging.warning(batch)
                samples.append(batch)
            # 将样本记录到日志中
            logging.info(f"Epoch {epoch} - First 5 samples: {samples}")
        
        
#         def _get_train_dataloader(self, train_dataset):
#             return DataLoader(
#                 train_dataset,
#                 batch_size=128,  
#                 shuffle = True
#             )
        
#         def get_train_dataloader(self) -> DataLoader:
#             """
#             Returns the training :class:`~torch.utils.data.DataLoader`.

#             Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
#             to distributed training if necessary) otherwise.

#             Subclass and override this method if you want to inject some custom behavior.
#             """
#             if self.train_dataset is None:
#                 raise ValueError("Trainer: training requires a train_dataset.")
#             train_sampler = self._get_train_sampler()

#             return DataLoader(
#                 self.train_dataset,
#                 batch_size=self.args.train_batch_size,
#                 shuffle = True, 
#                 sampler=train_sampler,
#                 collate_fn=self.data_collator,
#                 drop_last=self.args.dataloader_drop_last,
#                 num_workers=self.args.dataloader_num_workers,
#             )
        
    trainer = MyTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
    
if __name__ == "__main__":
    train()
