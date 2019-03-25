import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)
#tf.logging.set_verbosity(tf.logging.INFO)
# Other settings

tf.logging.debug('debug message...')
tf.logging.info('info message...')
tf.logging.warn('warn message')
tf.logging.error('error message')
tf.logging.fatal('fatal message')

class SkipRun(Exception):
  pass


for i in range(5):
    try:
        if i==2:
            raise SkipRun
        else:
            print(i)
    except Exception:
        print('Exception')
    except SkipRun:
        continue
    finally:
        print('Done.')


import logging

# logging.getLogger('tensorflow').propagate = False
# logging.getLogger('tensorflow').format = '%(message)s'
logging.basicConfig(level=logging.INFO)#, format='%(message)s')

logging.warning('warning...')

logger = logging.getLogger("new_logger")
logger.error("error message")





import contextlib

@contextlib.contextmanager
def tag(name):
    print("<%s>" % name)
    yield 'morning...'
    print("</%s>" % name)

with tag("h1") as q:
    print("good",q)


d = {'a':1,'b':2}

for key,values in  d.items():
    print (key,values)



import multiprocessing
import time


def proc1(pipe):
    while True:
        for i in range(100):
            print("proc1发送 %s" % i)
            pipe.send(i)
            time.sleep(2)


# def proc2(pipe):
#     while True:
#         print('proc2 接收:', pipe.recv())
#         #time.sleep(2)

def proc2(pipe):
    print('proc2 接收:', pipe.recv())
    print('proc2 接收:', pipe.recv())
    print('proc2 接收:', pipe.recv())
        #time.sleep(2)

# Build a pipe
pipe = multiprocessing.Pipe()
print(pipe)

# Pass an end of the pipe to process 1
p1 = multiprocessing.Process(target=proc1, args=(pipe[0],))
# Pass the other end of the pipe to process 2
p2 = multiprocessing.Process(target=proc2, args=(pipe[1],))

p1.start()
p2.start()
p1.join()
p2.join()