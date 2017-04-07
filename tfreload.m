clc,clear,close all;
clear all
clear classes
mod = py.importlib.import_module('tf');
py.importlib.reload(mod);
fprintf('py.tf reloaded , version = %d\n',double(py.tf.version));
clear all