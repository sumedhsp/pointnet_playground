from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from dataset import ShapeNetDataset, ModelNetDataset
from model import PointNetCls, feature_transform_regularizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=128, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=5000, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

from collections import defaultdict

class_correct = defaultdict(int)
class_total = defaultdict(int)
class_predicted = defaultdict(int)

classifier = classifier.eval()

total_correct = 0
total_testset = 0

for i, data in enumerate(testdataloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()    
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

    for t, p in zip(target, pred_choice):
        if t == p:
            class_correct[t.item()] += 1
        class_total[t.item()] += 1
        class_predicted[p.item()] += 1

print("Overall Accuracy {}".format(total_correct / float(total_testset)))

# Displaying per-class accuracy
for cls in class_total.keys():
    class_accuracy = class_correct[cls] / class_total[cls]
    print(f"Class {cls} Accuracy: {class_accuracy: .2f}")

# Precision, Recall, and F1-score for class 7
cls = 7
if cls in class_total and cls in class_predicted:
    tp = class_correct[cls]  # True Positives
    fn = class_total[cls] - tp  # False Negatives
    fp = class_predicted[cls] - tp  # False Positives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nClass {cls} Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score:.2f}")
else:
    print(f"\nClass {cls} not found in dataset.")



