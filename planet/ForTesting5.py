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


from planet import tools
ad = tools.AttrDict({'a':1})
ad1 = ad.copy()


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

