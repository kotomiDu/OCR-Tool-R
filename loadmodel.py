import tensorflow as tf
import os


tf.app.flags.DEFINE_string('tesseract_model_path', '/home/kotomi/tesstutorial/engoutput19', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', './testoutput', '')
tf.app.flags.DEFINE_string('words_to_detect', 'headshot', '')
tf.app.flags.DEFINE_bool('detect_word_flag', False, '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

FLAGS = tf.app.flags.FLAGS

sess =  tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
meta_path = os.path.join(FLAGS.checkpoint_path,'model.ckpt-49491.meta')
saver = tf.train.import_meta_graph(meta_path)
saver.restore(sess, model_path)


'''
load value of weight and bias
'''
f =  open("model.txt",'w') 
vars = tf.trainable_variables()
print(vars) #some infos about variables...
vars_vals = sess.run(vars)
for var, val in zip(vars, vars_vals):
    f.write("var: {}, value: {}\n".format(var.name, val))
f.close()


'''
visualize graph
'''
summary_writer = tf.summary.FileWriter('./',  tf.get_default_graph())
output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]

with open("model.txt",'w') as f:
    for i  in output_node_names:
        f.write(i+'\n')

'''
Freeze the graph
save tf model to pb file
'''
frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names)


with open('output_graph.pb', 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())







