from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
from transformers import Trainer, TrainingArguments, get_scheduler
import torch
from tqdm import tqdm
from sklearn.metrics import (ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt

def train(args, train_loader, test_loader, model):
    if not os.path.exists(args.load_dir):
        os.makedirs(args.load_dir)
    if args.model == ['roberta-base','bert-base']:
        train_huggingface(args)
    best_train_loss, best_test_loss = np.inf, np.inf
    print("Starting training...")
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    model = model.to(args.device)
    best_epoch = 0
    for epoch in range(args.epoches):
        loss_train, loss_test = train_epoch(args, model, train_loader, test_loader, optimizer, loss_func)
        print("Epoch {}: Training Loss {}  Testing loss {}".format(epoch, loss_train, loss_test))

        if loss_test <= best_test_loss:
            best_test_loss = loss_test
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.load_dir, 'best_model.pt'))

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
    }, os.path.join(args.load_dir, 'last_model.pt'))

def train_epoch(args, model, train_loader, test_loader, optimizer, loss_func):
    model.train()
    loss_train_tot = 0
    loss_test_tot = 0
    for data, label in tqdm(train_loader, total=len(train_loader)):
        data = data.to(args.device)
        label = label.to(args.device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        loss_train_tot += loss.item()

    model.eval()
    for data, label in tqdm(test_loader, total=len(test_loader)):
        data = data.to(args.device)
        label = label.to(args.device)
        pred = model(data)
        loss = loss_func(pred, label)
        loss_test_tot += loss.item()

    los_train_avg = loss_train_tot / len(train_loader)
    loss_test_avg = loss_test_tot / len(test_loader)
    
    return los_train_avg, loss_test_avg

def evaluate(args,test_loader,model):
    model = model.to(args.device)
    true_label = torch.tensor([])
    pred_label = torch.tensor([])
    for data, label in tqdm(test_loader, total=len(test_loader)):
        data = data.to(args.device)
        pred_label = torch.cat([pred_label, model(data).argmax(1).cpu()])
        true_label = torch.cat([true_label, label])
    ConfusionMatrixDisplay.from_predictions(true_label, pred_label)
    plt.show()

def train_huggingface(args, model, train_data, test_data):
    def compute_metrics(pred):
        labels = pred.label_ids
        if args.model == 'facebook/bart-base':
            preds = torch.tensor(pred.predictions[0]).argmax(-1)
        else:    
            preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir = args.load_dir,
        num_train_epochs=args.epoches,
        per_device_train_batch_size = args.batch_size,    
        per_device_eval_batch_size= args.batch_size,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        disable_tqdm = False, 
        warmup_steps=500,
        weight_decay=0.01,
        fp16 = True,
        logging_dir=args.load_dir,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        report_to=None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data
    )
    
    # train the model
    trainer.train()

    trainer.evaluate()

def train_bart(args, model, train_loader, test_loader):
    best_train_loss, best_test_loss = np.inf, np.inf
    print("Starting training...")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.epoches*len(train_loader)
    )
    loss_func = torch.nn.CrossEntropyLoss()
    model = model.to(args.device)
    best_epoch = 0
    for epoch in range(args.epoches):
        loss_train, loss_test = train_epoch_bart(args, model, train_loader, train_loader, optimizer, lr_scheduler, loss_func)
        print("Epoch {}: Training Loss {}/{}  Testing loss {}/{}".format(epoch, loss_train, len(train_loader), loss_test, len(test_loader)))

        # if loss_test <= best_test_loss:
        #     best_test_loss = loss_test
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), os.path.join(args.load_dir, 'best_model.pt'))

def train_epoch_bart(args, model, train_loader, test_loader, optimizer, lr_scheduler, loss_func):
    loss_train_tot = 0
    loss_test_tot = 0

    model.train()
    for data_infor in tqdm(train_loader, total=len(train_loader)):
        input_ids = torch.tensor(data_infor['input_ids'], device=args.device)
        attention_mask = torch.tensor(data_infor['attention_mask'], device=args.device)
        label = torch.tensor(data_infor['label'], device=args.device)
        if label.dim() == 0:
            label = label.unsqueeze(0).unsqueeze(0)
        elif label.dim() == 1:
            label = label.unsqueeze(0)
            
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        loss = model(input_ids=input_ids,attention_mask=attention_mask,labels=label,return_dict=True).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_train_tot += loss.item()

    model.eval()
    for data_infor in tqdm(test_loader, total=len(test_loader)):
        input_ids = torch.tensor(data_infor['input_ids'], device=args.device)
        attention_mask = torch.tensor(data_infor['attention_mask'], device=args.device)
        label = torch.tensor(data_infor['label'], device=args.device)
        if label.dim() == 0:
            label = label.unsqueeze(0).unsqueeze(0)
        elif label.dim() == 1:
            label = label.unsqueeze(0)
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        pred = model(input_ids=input_ids,labels=label,return_dict=True).logits
        loss = loss_func(pred, label[0])
        loss_test_tot += loss.item()

    los_train_avg = loss_train_tot / len(train_loader)
    loss_test_avg = loss_test_tot / len(test_loader)
    
    return los_train_avg, loss_test_avg