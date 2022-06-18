#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import collections
import re
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import jieba
import numpy as np
from datasets import load_dataset

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import (
    MT5Config,
    T5TokenizerFast,
    MBartConfig,
    MBart50TokenizerFast,
    MBartTokenizerFast,
    M2M100Config,
    M2M100Tokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import T5ForConditionalGeneration, MBartForConditionalGeneration, M2M100ForConditionalGeneration
from trainer import Seq2SeqTrainer

import logging
logger = logging.getLogger(__name__)


MODEL_CONFIG = {
    'mt5-small': [MT5Config, T5TokenizerFast],
    'mt5-base': [MT5Config, T5TokenizerFast],
    'mbart-25': [MBartConfig, MBartTokenizerFast],
    'mbart-50': [MBartConfig, MBart50TokenizerFast],
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."})
    data_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(default='train.jsonl', metadata={"help": "The input training data file (a jsonlines or csv file)."})
    validation_file: Optional[str] = field(
        default='dev.jsonl',
        metadata={"help": "An optional input evaluation data file to evaluate the metrics (rouge) on"
                  "(a jsonlines or csv file)."},
    )
    test_file: Optional[str] = field(
        default='test.jsonl',
        metadata={"help": "An optional input test data file to evaluate the metrics (rouge) on"
                  "(a jsonlines or csv file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
            "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help":
            "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                  "value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                  "value if set."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                  "value if set."},
    )
    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help":
            "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    predict_with_generate: bool = field(default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."})
    early_stopping_patience: int = field(default=-1, metadata={"help": "-1 means never early stop."})


# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

# Log on each process the small summary:
logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}," +
               f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")

# Set the verbosity to info of the Transformers logger (on main process only):
# if training_args.should_log:
transformers.utils.logging.set_verbosity_info()

logger.info(f"Data parameters {data_args}")
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets
data_files = {}
if training_args.do_train:
    train_file = os.path.join(data_args.data_path, data_args.data_name, data_args.train_file)
    data_files["train"] = train_file
if training_args.do_eval:
    validation_file = os.path.join(data_args.data_path, data_args.data_name, data_args.validation_file)
    data_files["validation"] = validation_file
if training_args.do_predict:
    test_file = os.path.join(data_args.data_path, data_args.data_name, data_args.test_file)
    data_files["test"] = test_file

data_path = os.path.join(data_args.data_path, "data", 'dataset.py')
cache_dir = os.path.join(data_args.data_path, data_args.data_name, 'cache')
raw_datasets = load_dataset(data_path, data_files=data_files, cache_dir=cache_dir)


def get_gen_kwargs(model_name, tokenizer, model):
    decoder_start_token_id, eos_token_id = None, None
    if 'mbart' in model_name:
        decoder_start_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.eos_token_id
    elif 't5' in model_name:
        decoder_start_token_id = model.config.decoder_start_token_id
        eos_token_id = tokenizer.eos_token_id
    elif 'm2m100' in model_name:
        decoder_start_token_id = model.config.decoder_start_token_id
        eos_token_id = tokenizer.eos_token_id

    assert decoder_start_token_id is not None and eos_token_id is not None
    gen_kwargs = {
        "max_length": data_args.val_max_target_length,
        "num_beams": data_args.num_beams,
        "early_stopping": True,
        "decoder_start_token_id": decoder_start_token_id,
        "eos_token_id": eos_token_id
    }

    logger.info(f"Train with model name: {model_name}\n\
        tokenizer: {tokenizer.__class__}\n\
        model: {model.__class__}\n\
        decoder_start_token: {tokenizer.convert_ids_to_tokens(decoder_start_token_id)}, eos_token: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
    return gen_kwargs

# Load pretrained model and tokenizer
config, tokenizer, model = None, None, None

if 't5' in model_args.model_name_or_path:
    config = MT5Config.from_pretrained(model_args.model_name_or_path)
    tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
elif 'mbart-25' in model_args.model_name_or_path:
    config = MBartConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = MBartTokenizerFast.from_pretrained(model_args.model_name_or_path)
    model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
elif 'mbart-large-50' in model_args.model_name_or_path:
    config = MBartConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_args.model_name_or_path)
    model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
elif 'm2m100' in model_args.model_name_or_path:
    config = M2M100Config.from_pretrained(model_args.model_name_or_path)
    tokenizer = M2M100Tokenizer.from_pretrained(model_args.model_name_or_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
# tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
# model = AutoModel.from_pretrained(model_args.model_name_or_path)

gen_kwargs = get_gen_kwargs(model_args.model_name_or_path, tokenizer, model)
assert tokenizer is not None and model is not None
model.resize_token_embeddings(len(tokenizer))

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
if training_args.do_train:
    column_names = raw_datasets["train"].column_names
elif training_args.do_eval:
    column_names = raw_datasets["validation"].column_names
elif training_args.do_predict:
    column_names = raw_datasets["test"].column_names
else:
    logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
    exit()

# Temporarily set max_target_length for training.
max_target_length = data_args.max_target_length
padding = False

if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning("label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                   f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

log_example = True

def preprocess_function(examples):
    # cross: {"src": "[\"再见\", \"再见\"]<Zh>", "tgt": "好 的 ， 再见 。"}
    #        {"src": "[\"Say goodbye\", \"Say goodbye\"] <En>", "tgt": "OK, goodbye."}
    # mono: {"src": "[\"寒暄\", \"寒暄\"]</s>[1] 你好 啊</s>你好 ！ <Zh>", "tgt": "近来 生活 怎么样 啊"}
    #       {"src": "[\"Greetings\", \"Greetings\"]</s>[1] Hello</s>Hello! <En>", "tgt": "How's life these days"}

    model_inputs = collections.defaultdict(list)
    for src_line, tgt_line in zip(examples['src'], examples['tgt']):
        if src_line[-1] == ">":
            src, lan = src_line[:-5], src_line[-4:]
        else:
            src, lan = src_line[:-3], src_line[-3:]

        # assert lan in ['<En>', '<Zh>']
        # lan_to_token = {'<En>': 'en_XX', '<Zh>': 'zh_CN'}
        lan_to_token = {lan: lan}
        
        src_tokens = tokenizer.tokenize(src, max_length=data_args.max_source_length, padding=padding, truncation=True)

        tgt = tgt_line
        tgt_tokens = tokenizer.tokenize(tgt, max_length=max_target_length, padding=padding, truncation=True)

        if tgt_tokens[-1] != tokenizer.eos_token:
            tgt_tokens.append(tokenizer.eos_token)

        if isinstance(tokenizer, MBartTokenizerFast):
            src_tokens = src_tokens + [tokenizer.eos_token, lan_to_token[lan]]
        elif isinstance(tokenizer, MBart50TokenizerFast):
            src_tokens = [lan_to_token[lan]] + src_tokens + [tokenizer.eos_token]
        elif isinstance(tokenizer, M2M100Tokenizer):
            pass
        else:
            src_tokens = src_tokens + tokenizer.tokenize(lan) + [tokenizer.eos_token]

        input_id = tokenizer.convert_tokens_to_ids(src_tokens)
        label = tokenizer.convert_tokens_to_ids(tgt_tokens)

        model_inputs["input_ids"].append(input_id)
        model_inputs["labels"].append(label)
        if isinstance(tokenizer, M2M100Tokenizer):
            lang_name = lan.lower().replace("<", "").replace(">", "").replace(" ", "")
            model.config.forced_bos_token_id = lang_name

    global log_example
    if log_example:
        i = 0
        example = {
            'src': examples['src'][i],
            'input_ids': model_inputs["input_ids"][i],
            'input_ids_deocde': tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][i]),
            'tgt': examples['tgt'][i],
            'labels': model_inputs["labels"][i],
            'labels_decode': tokenizer.convert_ids_to_tokens(model_inputs["labels"][i])
        }

        log_strs = ["*** Input Example ***"]
        for k, v in example.items():
            log_strs.append(k + ':\n  ' + str(v))
        logger.info('\n'.join(log_strs))

        log_example = False

    return model_inputs


if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    predict_dataset = predict_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on prediction dataset",
    )

log_feature = True

# Data collator
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        features["decoder_input_ids"] = decoder_input_ids

        global log_feature
        if log_feature:
            log_strs = ["*** Feature ***"]
            for k, v in features.items():
                log_strs.append(k + ':\n  ' + str(v[0]))
            logger.info('\n'.join(log_strs))

            log_feature = False

        return features


label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None
)


def postprocess_text(preds, refs):
    preds = [pred.strip() for pred in preds]
    refs = [ref.strip() for ref in refs]

    return preds, refs


from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sacrebleu

def compute_score(preds, refs):
    score = {}
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    score['bleu'] = bleu.score

    preds = [pred.split() for pred in preds]
    refs = [[ref.split()] for ref in refs]
    try:
        weights = []
        for n in [1, 2]:
            weights.append(tuple([1 / n] * n))

        bleu_score = corpus_bleu(refs, preds, weights=weights, smoothing_function=SmoothingFunction().method3)
    except ZeroDivisionError as _:
        logger.info('the bleu score is invalid')
        bleu_score = [0] * 2

    for i, n in enumerate([1, 2]):
        score[f'bleu-{n}'] = bleu_score[i] * 100

    return score


log_prediction = True


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    post_preds, post_labels = postprocess_text(decoded_preds, decoded_labels)

    # log prediction
    global log_prediction
    if log_prediction:
        i = 0
        example = {
            'predict_ids': preds[i].tolist(),
            'predict_ids_convert': tokenizer.convert_ids_to_tokens(preds[i]),
            'predict_ids_deocde': post_preds[i],
            'label_ids': labels[i].tolist(),
            'labels_ids_convert': tokenizer.convert_ids_to_tokens(labels[i]),
            'labels_ids_decode': post_labels[i]
        }

        log_strs = ["*** Prediction Example ***"]
        for k, v in example.items():
            log_strs.append(k + ':\n  ' + str(v))
        logger.info('\n'.join(log_strs))

        log_prediction = False

    result = compute_score(post_preds, post_labels)

    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    return result

if training_args.early_stopping_patience > 0:
    es_callback = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)
else:
    es_callback = None

# Initialize our Trainer
trainer = Seq2SeqTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset if training_args.do_train else None,
                         eval_dataset=eval_dataset if training_args.do_eval else None,
                         tokenizer=tokenizer,
                         data_collator=data_collator,
                         callbacks=[es_callback] if es_callback else None,
                         compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                         gen_kwargs=gen_kwargs)

# Training
if training_args.do_train:
    logger.info("*** Train ***")
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    metrics = {k: round(v, 2) for k, v in metrics.items()}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(metric_key_prefix="evaluate")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    metrics = {k: round(v, 2) for k, v in metrics.items()}
    trainer.log_metrics("evaluate", metrics)
    trainer.save_metrics("evaluate", metrics)

if training_args.do_predict:
    logger.info("*** Predict ***")

    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
    metrics = predict_results.metrics
    max_predict_samples = (data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset))
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    metrics = {k: round(v, 2) for k, v in metrics.items()}
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
