	�� Z+��?�� Z+��?!�� Z+��?	�俻�@�俻�@!�俻�@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�� Z+��?ș&l?�?A* �3h��?Y<�)�?*	��(\�Bu@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?��I�?!�hk��J@)N{JΉ=�?1��Um6�I@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map���&�+�?!-�����A@)�=�Е�?1[swp�;<@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat0,�-X�?!�[���@@)�H��Q,�?1/�t�j�@:Preprocessing2F
Iterator::Model���0�?!kf��/@)�n��;�?1^���@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�9�!�?!�c΁�@)a���)�?13��W�@:Preprocessing2U
Iterator::Model::ParallelMapV2��jQL~?!y}�I�e@)��jQL~?1y}�I�e@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchBA)Z�x?!r��7���?)BA)Z�x?1r��7���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6t�?Pn�?!E�!l)4M@)K�|%�r?1�-�(Q�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9~�4bfo?!Ka�5w�?)9~�4bfo?1Ka�5w�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range��6�^i?!|v�9p"�?)��6�^i?1|v�9p"�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::TensorSlicepD��k�\?!���M{�?)pD��k�\?1���M{�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::TensorSliceJ]2���A?!�2P��?)J]2���A?1�2P��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t33.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�俻�@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ș&l?�?ș&l?�?!ș&l?�?      ��!       "      ��!       *      ��!       2	* �3h��?* �3h��?!* �3h��?:      ��!       B      ��!       J	<�)�?<�)�?!<�)�?R      ��!       Z	<�)�?<�)�?!<�)�?JCPU_ONLYY�俻�@b 