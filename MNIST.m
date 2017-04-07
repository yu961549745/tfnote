classdef MNIST < handle
    properties
        tf
    end
    methods
        function s = MNIST(meta,ckp)
            s.tf=py.tf.MNIST(meta,ckp);
        end
        function delete(s)
            s.tf.close();
        end
        function y = predict(s,img)
            if iscell(img)
                x=cell2py(img);
            else
                x=mat2py(img);
            end
            y=s.tf.predict(x);
            y=nparray2mat(y);
        end
    end
end
function y = cell2py(x)
y=cellfun(@(t){mat2py(t)'},x(:));
y=cell2mat(y)';
end
function y = mat2py(x)
n=ndims(x);
if n>=2
    id=1:n;
    id(1:2)=[2,1];
    y=permute(x,id);
end
y=y(:)';
end
function y = nparray2mat(x)
y=cell(x.tolist());
y=cellfun(@double,y);
end