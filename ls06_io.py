""" 
TensorFlow 模型导入导出 

MetaGraph 包含 GraphDef 和相关的 meatadata ，可以用于保存 graphs 。
可以用于继续训练，和执行。包含以下几个部分：
MetaInfoDef metadata，例如版本信息
GraphDef 描述  graph
SaverDef 描述 saver
CollectionDef 描述变量等信息

采用 saver.export_meta_graph(filename=None, collection_list=None, as_text=False) 可以导出整个 graph
如果不指定 collection_list 则会导出所有  collection
MetaGraph 也会被 tf.train.Saver 的 save 函数自动导出

"""
