import tensorflow as tf

summary = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

optimizer = 3
feed_dict=3
global_step=4

writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)

s,_ = sess.run([summary, optimizer],feed_dict = feed_dict)
writer.add_summary(s,global_step=global_step)
global_step += 1

writer = tf.summary.FileWriter("./logs/xor_logs")

