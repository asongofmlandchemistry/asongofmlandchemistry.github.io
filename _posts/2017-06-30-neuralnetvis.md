---
title: "Neural Networks: Why thou so Powerful!"
date: 2017-06-30
tags: [neural networks, decison boundaries, activation functions, piecewise linear]
header:
    image: "/images/header.jpg"
---

Hi guys, Today's blog will be a little different from my other blogs, today we won't learn how to do something cool with neural nets, but instead we will see what makes Neural Networks so great for machine learning problems. This post is for the people who know what neural nets are but find it difficult to understand why they work so well. I will take an example of binary classification to demonstrate how neural nets are able to approximate highly complex functions accurately. So lets dive into the details, its going to be fun.

## Binary Classification
![alt text](/images/neuralnets/data.png)

As you can see from the figure we are given a set of points and some are yellow and other are blue demostrating the different classes(or categories) to which the points belong. Our job is to learn from this data so that given a new point we can predict its class accurately.

The simplest approach to this problem is using a Linear Classifier. What a Linear Classifier does is, it learns a line (known as a decision boundary) which seperates the data points, such that points lying on one side belongs to one class and points lying on the other side belongs to the other class. Like in our problem, what a linear classifier will do is find a line which separates the blue and yellow points.

Now as we can see from the plot, it is impossible to find a line which can separate these points perfectly. I trained a softmax classifier on these data points and got the following plot.
![alt text][/images/neuralnets/myfig.png]