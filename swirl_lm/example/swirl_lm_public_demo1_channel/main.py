from absl import app
from absl import flags
from cloud_tpu_client import Client
import tensorflow as tf

_TPU = flags.DEFINE_string('tpu', None, 'TPU id.', required=True)
_ZONE = flags.DEFINE_string('zone', None, 'Zone.', required=True)
_PROJECT = flags.DEFINE_string('project', None, 'Cloud project id.', 
                               required=True)

def main(unused_argv):
  c = Client(tpu=_TPU.value, zone=_ZONE.value, project=_PROJECT.value)

  c.configure_tpu_version('tpu-vm-tf-2.9.1-pod', restart_type='ifNeeded')
  c.wait_for_healthy()

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=_TPU.value, zone=_ZONE.value, project=_PROJECT.value
  ) 
  print(tf.config.list_logical_devices('TPU') )


if __name__ == '__main__':
  app.run(main)


