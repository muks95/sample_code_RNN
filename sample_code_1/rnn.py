import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab
import variation_sentence as vs
from sklearn.manifold import TSNE
import gensim
RESET_AFTER = 50
class Config(object):
    embed_size = 35
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs =30
    lr = 0.03
    l2 = 0.04
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)

class RNN_Model():

    def load_data(self):
        self.train_data, self.dev_data, self.test_data,self.sample_data = tr.simplified_data(20, 100, 200)
        self.final_W=[]
        self.final_U=[]
        self.final_bias=[]
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def inference(self, tree, predict_only_root=False):
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.iteritems()]
            node_tensors = tf.concat(0, node_tensors)
            
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        with tf.variable_scope('Composition',reuse=None) as scope:
            
            W1=tf.get_variable("W1",[2*self.config.embed_size,self.config.embed_size])
            b1=tf.get_variable("B1",[1,self.config.embed_size])
            self.X=self.model.syn0
            embedding=tf.get_variable("embedding",self.X.shape)
            scope.reuse_variables()
        with tf.variable_scope('Projection',reuse=None) as scope1:
            U=tf.get_variable("U",[self.config.embed_size,1])
            scope1.reuse_variables()

    def add_model(self, node):
        with tf.variable_scope('Composition', reuse=True):
            embedding=tf.get_variable("embedding")  
            W1=tf.get_variable("W1")
            b1=tf.get_variable("B1")

        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf:
            index=self.vocab.encode(node.word)
            curr_node_tensor=tf.expand_dims(tf.gather(embedding,index),0)

        elif node.dropout:
            return node_tensors
        elif node.left.dropout:
            node_tensors.update(self.add_model(node.right))
            curr_node_tensor=node_tensors[node.right]
        elif node.right.dropout:
            node_tensors.update(self.add_model(node.left))
            curr_node_tensor=node_tensors[node.left]
        else:
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))

            W_split=tf.split(0,2,W1)
            W_left=W_split[0]
            W_right=W_split[1]
            node_left_tesnor=node_tensors[node.left]
            node_right_tensor=node_tensors[node.right]
            temp=tf.matmul(node_left_tesnor,W_left) + tf.matmul(node_right_tensor,W_right) + b1
            curr_node_tensor=tf.maximum(temp,0)

        node_tensors[node] = curr_node_tensor
        return node_tensors
    def add_model_greedy(self,new_model,sentence):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.add_model_vars()
            if new_model:
                init = tf.global_variables_initializer()
                #sess.run(init)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, './weights/%s.temp'%self.config.model_name)
           
            with tf.variable_scope('Composition',reuse=True):
                embedding=tf.get_variable("embedding")
                W1=tf.get_variable("W1")
                b1=tf.get_variable("B1")
            with tf.variable_scope("Projection",reuse=True):
                U=tf.get_variable("U")

                W_split=tf.split(0,2,W1)
                W_left=sess.run(W_split[0])
                W_right=sess.run(W_split[1])
                #filename='W_right_check_greedy.txt'
                #np.savetxt(filename,W_right)
                bias=sess.run(b1)
                U_weight=sess.run(U)
                temp=[]
                temp1=[]
                for word in sentence:
                    index=self.vocab.encode(word)
                    curr_node_tensor=tf.expand_dims(tf.gather(embedding,index),0)
                    temp.append(sess.run(curr_node_tensor))
                    temp1.append("(2 " + word + ")")
                t=len(temp)
                while(1):
                    index =0
                    parent=np.ndarray(shape=(self.config.embed_size),dtype=float)
                    cur=0
                    for i in range(t):
                        j=i+1
                        if(j>=t):
                            break
                        temp2=np.matmul(temp[i],W_left) + np.matmul(temp[j],W_right) + bias
                        temp3=np.maximum(temp2,0)
                        score=np.matmul(temp3,U_weight) 
                        if(score[0] > cur):
                            cur=score
                            parent=temp3
                            index=i
                    del temp[index]
                    del temp[index]
                    X ="(2 " + temp1[index] + " " + temp1[index+1] +" )"
                    del temp1[index]
                    del temp1[index]
                    temp1.insert(index,X)
                    temp.insert(index,parent)
                    t=len(temp)
                    if(t==1):
                        return tr.Tree(X)
    def get_rep(self,new_model,sentence):
        #with tf.Graph().as_default(), tf.Session() as sess:
        #    self.add_model_vars()
        
        with tf.Graph().as_default(), tf.Session() as sess:
            self.add_model_vars()
            if new_model:
                init = tf.global_variables_initializer()
                #sess.run(init)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, './weights/%s.temp'%self.config.model_name)
            with tf.variable_scope('Composition',reuse=True):
                embedding=tf.get_variable("embedding")
                W1=tf.get_variable("W1")
                b1=tf.get_variable("B1")
            with tf.variable_scope("Projection",reuse=True):
                U=tf.get_variable("U")
                W_split=tf.split(0,2,W1)
                W_left=sess.run(W_split[0])
                W_right=sess.run(W_split[1])
                #filename='W_right_check_rep.txt'
                #np.savetxt(filename,W_right)
                bias=sess.run(b1)
                U_weight=sess.run(U)
                temp=[]
                temp1=[]
                for word in sentence:
                    index=self.vocab.encode(word)
                    curr_node_tensor=tf.expand_dims(tf.gather(embedding,index),0)
                    temp.append(sess.run(curr_node_tensor))
                    temp1.append("(2 " + word + ")")
                t=len(temp)
                while(1):
                    index =0
                    parent=np.ndarray(shape=(self.config.embed_size),dtype=float)
                    cur=0
                    for i in range(t):
                        j=i+1
                        if(j>=t):
                            break
                        temp2=np.matmul(temp[i],W_left) + np.matmul(temp[j],W_right) + bias
                        temp3=np.maximum(temp2,0)
                        score=np.matmul(temp3,U_weight) 
                        if(score[0] > cur):
                            cur=score
                            parent=temp3
                            index=i
                    del temp[index]
                    del temp[index]
                    X ="(2 " + temp1[index] + " " + temp1[index+1] +" )"
                    del temp1[index]
                    del temp1[index]
                    temp1.insert(index,X)
                    temp.insert(index,parent)
                    t=len(temp)
                    if(t==1):
                        return parent        
            
    def add_projections(self, node_tensors):
        logits = None

        with tf.variable_scope("Projection",reuse=True):
            U=tf.get_variable("U")
        logits=tf.matmul(node_tensors,U) 

        return logits

    def loss(self, logits, logits1):
        loss = None

        with tf.variable_scope("Composition",reuse=True):
            W1=tf.get_variable("W1")
            b1=tf.get_variable("B1")
            l2_loss=tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
            tf.add_to_collection(name="l2_loss",value=l2_loss)
        l2_loss=self.config.l2 * tf.get_collection("l2_loss")[0]
        objective_loss = tf.maximum(tf.reduce_sum(logits) - tf.reduce_sum(logits1),0)
        loss=objective_loss + l2_loss

        return loss
    def load_data1(self):
        train_data,dev_data,test_data,sample_data = tr.simplified_data(700, 100, 200)
        vocab = Vocab()
        train_sents = [t.get_words() for t in train_data]
        vocab.construct(list(itertools.chain.from_iterable(train_sents)))
        return train_sents  
    def training(self, loss):
        train_op = None

        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        train_op=optimizer.minimize(loss)


        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data()

    def initalize(self):
        train_sents=self.load_data1()
        wordVocab=[]
        wv=[]
        for sentence in train_sents:
            for words in sentence:
                if words in wordVocab:
                    continue
                wordVocab.append(words)
        self.model = gensim.models.Word2Vec(train_sents,size=self.config.embed_size,min_count=1)    
    def run_epoch(self, new_model = False, verbose=True):
        step = 0
        loss_history = []
        X=len(self.train_data)
        if new_model:
            self.initalize()
        while step < X:
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    with tf.variable_scope('Composition', reuse=True):
                        embedding=tf.get_variable("embedding")
                        sess.run(embedding.assign(self.X))
                    saver = tf.train.Saver()
                    if not os.path.exists("./weights"):
                    	os.makedirs("./weights")
                    saver.save(sess, './weights/%s.temp'%self.config.model_name)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                for _ in xrange(RESET_AFTER):
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step]
                    sentence=tree.get_words()
                    #print sentence
                    '''
                    with tf.variable_scope("Composition",reuse=True):
                        W1=tf.get_variable("W1")
                        W_split=tf.split(0,2,W1)
                        W_left=sess.run(W_split[0])
                        W_right=sess.run(W_split[1])
                        filename='W_right_check_bgreedy.txt'
                        np.savetxt(filename,W_right)
                    '''

                    tree1=self.add_model_greedy(False,sentence) 
                    logits = self.inference(tree)
                    logits1 = self.inference(tree1)  
                    loss = self.loss(logits, logits1)
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    '''
                    with tf.variable_scope("Composition",reuse=True):
                        W1=tf.get_variable("W1")
                        W_split=tf.split(0,2,W1)
                        W_left=sess.run(W_split[0])
                        W_right=sess.run(W_split[1])
                        filename='W_right_check_agreedy.txt'
                        np.savetxt(filename,W_right)
                    #self.get_rep(sess)
                    '''
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name)

        return  loss_history
    
    def plot_with_labels(self,low_dim_embs, labels, filename='tsne2.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
            plt.savefig(filename)

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in xrange(self.config.max_epochs):
            print 'epoch %d'%epoch
            if epoch==0:
                 loss_history = self.run_epoch(new_model=True)
            else:
                 loss_history = self.run_epoch()
            #print loss_history
            complete_loss_history.extend(loss_history)
            epoch_loss = np.mean(loss_history)
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
            prev_epoch_loss = epoch_loss
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break

            #self.display(epoch)

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()
            
            
            
            X=len(self.sample_data)
            i=0
            rep=[]
            indx=[]
            while i<X:
                tree=self.sample_data[i]
                parent=self.get_rep(False,tree.get_words())
                rep.append(np.squeeze(parent))
                i=i+1
                sentence=tree.get_words()
                words=" ".join(sentence)
                words=unicode(words, 'utf-8')
                indx.append(words)
            final_indx=np.asarray(indx)
            final_rep=np.squeeze(np.asarray(rep))
       
            final_rep=np.nan_to_num(final_rep)
                #print final_rep.shape
            filename='sample_rep_epoch' + str(epoch) + '.png'
            tsne=TSNE(n_components=2,random_state=0)
            np.set_printoptions(suppress=True)
            Y=tsne.fit_transform(final_rep)
            self.plot_with_labels(Y,final_indx,filename)
        #print final_rep.shape
            
        print '\n\nstopped at %d\n'%stopped
        return {
            'loss_history': complete_loss_history,
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in itertools.izip(labels, predictions):
            confmat[l, p] += 1
        return confmat

def test_RNN():
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print 'Training time: {}'.format(time.time() - start_time)
    '''
    plt1.plot(stats['loss_history'])
    plt1.title('Loss history')
    plt1.xlabel('Iteration')
    plt1.ylabel('Loss')
    plt1.savefig("loss_history_sample.png")
    plt1.show()    
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    wv=np.asarray(model.final_W)
    Y = tsne.fit_transform(wv)
    plt.scatter(Y[:,0],Y[:,1])  

    plt.show()  
    '''
if __name__ == "__main__":
        test_RNN()
