clc,clear,close all;
% pyversion /Users/yjt/anaconda/bin/python
data=py.data.data;
test=cell(data.test.images.tolist());
test=cellfun(@(x){reshape(cell2mat(cell(x)),[28,28])'},test);
