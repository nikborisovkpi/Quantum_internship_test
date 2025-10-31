import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support


class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]

        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=False
        )

        labels = []
        word_ids = encoding.word_ids() # use word_ids to align labels
        previous_word_id = None

        # Word labeling logic
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                label = word_labels[word_id]
                if label == "O":
                    labels.append(self.label2id[label])
                else:
                    # If token is part of same word, put I-MOUNTAIN 
                    if word_id != previous_word_id:
                        labels.append(self.label2id["B-MOUNTAIN"])
                    else:
                        labels.append(self.label2id["I-MOUNTAIN"])
                previous_word_id = word_id

        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(labels)
        return encoding



def load_synthetic_data(file_path):
    texts, labels_list = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        text, labels = [], []
        for line in f:
            line = line.strip()
            if not line:  # Пустая строка означает конец текста
                if text and labels:
                    texts.append(" ".join(text))
                    labels_list.append(labels)
                text, labels = [], []
            else:
                word, label = line.split()
                text.append(word)
                labels.append(label)
    return texts, labels_list


def compute_metrics(eval_pred):
    # Compute main metrics
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels, pred_labels = [], []
    for pred, lab in zip(predictions, labels):
        for p, l in zip(pred, lab):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main(args):
    # Загрузка данных из файла
    train_texts, train_labels = load_synthetic_data(args.train_path)
    val_texts, val_labels = load_synthetic_data(args.val_path)

    # Put labels
    label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
    label2id = {label: i for i, label in enumerate(label_list)}

    # Load tokkenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id=label2id
    )

    # Create datasets
    train_dataset = NERDataset(train_texts, train_labels, tokenizer, label2id)
    val_dataset = NERDataset(val_texts, val_labels, tokenizer, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",             # metrics after every epoch
        save_strategy="no",                # do not save checkpoints
        logging_strategy="epoch",           
        logging_dir="./logs",               
        load_best_model_at_end=False,
        report_to="none",                   
    )


    # Trainer with metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Model training 
    trainer.train()

    # Model saving
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='train_data.txt')
    parser.add_argument('--val_path', type=str, default='val_data.txt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./ner_model')
    args = parser.parse_args()
    main(args)
