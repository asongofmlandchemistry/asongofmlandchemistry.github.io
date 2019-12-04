---
layout: externalpost
date: 2019-10-04
redirect_url: https://medium.com/swlh/how-to-use-pytorch-dataloaders-to-work-with-enormously-large-text-files-bbd672e955a0
title: How to use Pytorch Dataloaders to work with enormously large text files
tags: [pytorch, dataset, iterabledataset, dataloader, textfiles, machine learning]
---

Pytorch’s Dataset and Dataloader classes provide a very convenient way of iterating over a dataset while training your machine learning model. The way it is usually done is by defining a subclass of the PyTorch's Dataset class and then wrapping an object of it using a dataloader. This dataloader is then used to sample data from the dataset while training. However, this approach might require you to store the complete data in the memory (you can get away with it by utilizing hdf5 files) which makes it infeasible to work with when we have to read data from very big files.