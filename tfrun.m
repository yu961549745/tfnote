clc,clear,close all;clc;
%% read data
fprintf('loading mnist data...\n');
tic
if ~exist('mnist.mat','file')
    data=py.data.data;
    getImgs=@(x)cellfun(@(x){reshape(cell2mat(cell(x)),[28,28])'},...
        cell(x.images.tolist()));
    test=getImgs(data.test);
    train=getImgs(data.train);
    valid=getImgs(data.validation);
    save mnist test train valid
else
    load mnist test train valid
end
toc

%% loading tensorflow model
fprintf('loading TensorFlow model...\n');
tic
mnist=MNIST('MNIST_conv/conv.meta','MNIST_conv/conv-19999');
toc

%% predict
data=test(randperm(length(test),50));
fprintf('predicting...\n');
tic
res=mnist.predict(data);
toc
img=catImages(data,10);
imagesc(img);
colormap gray
axis equal
axis tight
axis off
title(num2str(res))
