clc,clear,close all;
clear all
clear classes
mod = py.importlib.import_module('tf');
py.importlib.reload(mod);
disp(py.tf.version)