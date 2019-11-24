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

![alt text](/images/neuralnets/myfig.png)
As we can see from the plot, the decision boundary is not able to separate the points belonging to different classes properly, there are many yellow points in the region of blue and vice versa. Hence a linear classifier isn't a good choice for solving this problem. Neural Networks however can solve this problem with much higher accuracy.There are two ways to think why they perform so well. Lets see them.

## First Idea: Making the data linearly separable
As we saw above, a linear function wasn't able to separate the data points we had, but what if I transform these points in such a way that they become linearly separable? We can think of neural networks as doing the same thing.
There are two important components of a neural network, a) Hidden Layers b) Hidden units in each layer. We call these hidden units as 'neurons'.Below is a figure comparing architectures of a linear classifier and a neural network.

                                  </p>
                                  <div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile userAgent=\&quot;Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36\&quot; version=\&quot;7.1.8\&quot; editor=\&quot;www.draw.io\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;9f9ec519-4e2e-932b-25ee-d0e356dcaa15\&quot; name=\&quot;Page-1\&quot;&gt;7Vxbd6M2EP41fuwekJCAxyS77fac7eWcPLR91BrZ5gSDCzi2++srjLBhhqQx5qK42YcsjEBC3zczmtHFM/qw3v+Uis3qlySQ0YxYwX5GP88I8Vyi/haCQymgDi8FyzQMSpF9FjyG/0gttLR0GwYyazyYJ0mUh5umcJ7EsZznDZlI02TXfGyRRM1WN2IpkeBxLiIs/SMM8pWW2tw/F3yV4XKlm/aIWxZ8F/OnZZpsY93ejNDF8V9ZvBZVXbqj2UoEya4mol9m9CFNkry8Wu8fZFRAW8FWvvfjC6Wn705lnL/lBc3Ts4i2uut7XUeWHyo01CsKeHVzv1uFuXzciHlRslPUK9kqX0fqzlaXItuUbCzCvVQt3OvaZZrL/YtfaJ/6rdRJJmuZpwf1iH7B10hpTTqpyO7Mi6tFqxojlUxoTVieKj6joS40IO3gUAwOMRocSscDhyNwMDRxcFcYo7qLk1g24SiflgEyxP/se61zrKVzlSyVkcjD52b1bT3WLfyehKrhE7Q2B4rHAGhZsk3nUr9VNzFQEbUASRaoKBfpUuaoIgWdONQe2xQPZIiiEzBvYs0diTXlr4+90u7RJCKpNzKRHWjyEE0HsxwPxIJw7Hm8Fk4h9l08j9+iwzxSDdwH4bO6XObHPpaiRXKk8Iwb/3ubVAU/ZMfI4049QNhmfy6savkWxlKkVV3qw8rqmk0ocUvDvX/LQySyLFyE8tLvAXqjSM+bCpLlafIkH5IoSc9GvwijCIhEFC5jdTtXSqK+gt4XKhSqqOlOF6zDIIhe0sgetA75ZB9rnd+idbQHraviDoNDJUYZGLIYgoe3wMP7gKclkjQsWILw0DHhcUYaeKcaZZkPdM9j3UZZlwAbt0FF/Y2yVc2dSZH7MP+zdv2XurY+sYvipBKXuo+ZikIO4xvWF4V8OApxpHRbdgVJ6c2uKBmOlLYI7SK7ilVbhWFZ1U0Xy6pnIMS5TRLdwUgkON4ZxrIAUewmiXKc4YiyEVHCNiwyhXDYHg69XqP1mtCr0tsGPoaFpmjEdEbEB09zCpuajQ+1RsQHh+6iCueNxYePiA+OokUVWZuKj9MyUT4YPtfOlA8QDRmVKtKugyx3QEV0uEH22onzAUgkU5KIksXOkRIFQx+xBiNxrGQREEVvkyg4l98jUR8J5GAk2p+4+ipuew6xHNdtVut4nzzOKWMeo5bNGR+K4UqXBmG492k50qIP0+apNtAHvyejdoZb06Q4T72Q8p5pnTYKcl3gTuE891sp9KFfhrrQI4Xkg8I6hR4MXXqiELn3HinE2f5lFL6VKGPsCm52MtGurl0d7N2uJs0tkF113bEzpl1Nv5ZoisVRuN3CRLqunpTp2+ImTRKhxVHnHTjNq6dkbsfi/HdA17WTL71b3KQZPbQ4lHiZ6DSvnpa5GYtzmGe8xVU+3ByLm3TOBFlc1+1NI1qcQxCF4oSsIWtrPly7rkylRmu1UFOnFe6I6LK25rSsXbdp+aT4VHvUXtl2OtTaI8P6c5kLMN0v+7y5BKiMs5tR2xYDVj3cYiK79TkYyAp9F6zgSRjDDt3YFhx5RtwGw/AEx8/xZluQ9U0cikMpAKqpzpkUR2L0ieICzF5ODoCNBraPcXdacO/j3AnDedNX1X0Zn4G3cERwQ+C7YO3TZji+aDtqBvcRdAIfZzwIfOy7bwh836LN8ZVizR8KfI7zl9+2+f/E5ShfDyIbTkfzObwtQLn6WKNNXjtiaX2cbyS+07Q2SrCrG+p8I8fjeys1hqA9iMk1BxrmvA39Xnxd23oEF+uiq/H3rPivKI70mPOr3KYiOl7kuyR9MpaWIExVGBwmxYs7meX9MMVBVsBaQgK77YcvOlClbs8/x1KmFeefvKFf/gU=&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://www.draw.io/js/viewer.min.js"></script>
                                  <p>