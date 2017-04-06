clc,clear,close all;
fid=fopen('1.txt');
x=textscan(fid,'%f');
x=x{1};
fclose(fid);

img=reshape(x,[28,28])';
disp(py.tf.version);
mnist=py.tf.MNIST();
y=mnist.predict(x(:)');
cellfun(@double,cell(y.tolist()))