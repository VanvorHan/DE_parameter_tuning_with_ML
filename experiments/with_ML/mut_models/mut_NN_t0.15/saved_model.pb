??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
??*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:?*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	?d*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:d*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:d2*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:2*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:2*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_11/kernel/m
?
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:d2*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:2*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameAdam/dense_11/kernel/v
?
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:d2*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:2*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:2*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?-B?- B?-
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemLmMmNmOmPmQmRmSvTvUvVvWvXvYvZv[
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
D
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

J	variables
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_4Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_56999275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst"/device:CPU:0*0
dtypes&
$2"	
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOpAssignVariableOpdense_10/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_1AssignVariableOpdense_10/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_2AssignVariableOpdense_11/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_3AssignVariableOpdense_11/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_4AssignVariableOpdense_12/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_5AssignVariableOpdense_12/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_13/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_13/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0	*
_output_shapes
:
Y
AssignVariableOp_8AssignVariableOp	Adam/iter
Identity_9"/device:CPU:0*
dtype0	
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_9AssignVariableOpAdam/beta_1Identity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_10AssignVariableOpAdam/beta_2Identity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_11AssignVariableOp
Adam/decayIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_12AssignVariableOpAdam/learning_rateIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_13AssignVariableOptotalIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_14AssignVariableOpcountIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_15AssignVariableOptotal_1Identity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_16AssignVariableOpcount_1Identity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_17AssignVariableOpAdam/dense_10/kernel/mIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_18AssignVariableOpAdam/dense_10/bias/mIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_19AssignVariableOpAdam/dense_11/kernel/mIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_20AssignVariableOpAdam/dense_11/bias/mIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_21AssignVariableOpAdam/dense_12/kernel/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_22AssignVariableOpAdam/dense_12/bias/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_23AssignVariableOpAdam/dense_13/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_24AssignVariableOpAdam/dense_13/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_25AssignVariableOpAdam/dense_10/kernel/vIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_26AssignVariableOpAdam/dense_10/bias/vIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_27AssignVariableOpAdam/dense_11/kernel/vIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_28AssignVariableOpAdam/dense_11/bias/vIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_29AssignVariableOpAdam/dense_12/kernel/vIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_30AssignVariableOpAdam/dense_12/bias/vIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_31AssignVariableOpAdam/dense_13/kernel/vIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_32AssignVariableOpAdam/dense_13/bias/vIdentity_33"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_34Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??

?-
?
#__inference__wrapped_model_56998037
input_4H
4sequential_3_dense_10_matmul_readvariableop_resource:
??D
5sequential_3_dense_10_biasadd_readvariableop_resource:	?G
4sequential_3_dense_11_matmul_readvariableop_resource:	?dC
5sequential_3_dense_11_biasadd_readvariableop_resource:dF
4sequential_3_dense_12_matmul_readvariableop_resource:d2C
5sequential_3_dense_12_biasadd_readvariableop_resource:2F
4sequential_3_dense_13_matmul_readvariableop_resource:2C
5sequential_3_dense_13_biasadd_readvariableop_resource:
identity??,sequential_3/dense_10/BiasAdd/ReadVariableOp?+sequential_3/dense_10/MatMul/ReadVariableOp?,sequential_3/dense_11/BiasAdd/ReadVariableOp?+sequential_3/dense_11/MatMul/ReadVariableOp?,sequential_3/dense_12/BiasAdd/ReadVariableOp?+sequential_3/dense_12/MatMul/ReadVariableOp?,sequential_3/dense_13/BiasAdd/ReadVariableOp?+sequential_3/dense_13/MatMul/ReadVariableOp?
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_3/dense_10/MatMulMatMulinput_43sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+sequential_3/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
sequential_3/dense_11/MatMulMatMul(sequential_3/dense_10/Relu:activations:03sequential_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
sequential_3/dense_11/BiasAddBiasAdd&sequential_3/dense_11/MatMul:product:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d|
sequential_3/dense_11/ReluRelu&sequential_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
sequential_3/dense_12/MatMulMatMul(sequential_3/dense_11/Relu:activations:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2|
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
sequential_3/dense_13/MatMulMatMul(sequential_3/dense_12/Relu:activations:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_3/dense_13/SoftmaxSoftmax&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_3/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp,^sequential_3/dense_11/MatMul/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_11/MatMul/ReadVariableOp+sequential_3/dense_11/MatMul/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
??
?
/__inference_sequential_3_layer_call_fn_56998957
input_4;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp?:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp?:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp?:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0}
dense_10/MatMulMatMulinput_4&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
*dense_10/dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(dense_10/dense_10/kernel/Regularizer/AbsAbs?dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_10/dense_10/kernel/Regularizer/SumSum,dense_10/dense_10/kernel/Regularizer/Abs:y:05dense_10/dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_10/dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_10/dense_10/kernel/Regularizer/mulMul3dense_10/dense_10/kernel/Regularizer/mul/x:output:01dense_10/dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_10/dense_10/kernel/Regularizer/addAddV23dense_10/dense_10/kernel/Regularizer/Const:output:0,dense_10/dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+dense_10/dense_10/kernel/Regularizer/SquareSquareBdense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_10/dense_10/kernel/Regularizer/Sum_1Sum/dense_10/dense_10/kernel/Regularizer/Square:y:05dense_10/dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_10/dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_10/dense_10/kernel/Regularizer/mul_1Mul5dense_10/dense_10/kernel/Regularizer/mul_1/x:output:03dense_10/dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_10/dense_10/kernel/Regularizer/add_1AddV2,dense_10/dense_10/kernel/Regularizer/add:z:0.dense_10/dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????do
*dense_11/dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
(dense_11/dense_11/kernel/Regularizer/AbsAbs?dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_11/dense_11/kernel/Regularizer/SumSum,dense_11/dense_11/kernel/Regularizer/Abs:y:05dense_11/dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_11/dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_11/dense_11/kernel/Regularizer/mulMul3dense_11/dense_11/kernel/Regularizer/mul/x:output:01dense_11/dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_11/dense_11/kernel/Regularizer/addAddV23dense_11/dense_11/kernel/Regularizer/Const:output:0,dense_11/dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
+dense_11/dense_11/kernel/Regularizer/SquareSquareBdense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_11/dense_11/kernel/Regularizer/Sum_1Sum/dense_11/dense_11/kernel/Regularizer/Square:y:05dense_11/dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_11/dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_11/dense_11/kernel/Regularizer/mul_1Mul5dense_11/dense_11/kernel/Regularizer/mul_1/x:output:03dense_11/dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_11/dense_11/kernel/Regularizer/add_1AddV2,dense_11/dense_11/kernel/Regularizer/add:z:0.dense_11/dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2o
*dense_12/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
(dense_12/dense_12/kernel/Regularizer/AbsAbs?dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_12/dense_12/kernel/Regularizer/SumSum,dense_12/dense_12/kernel/Regularizer/Abs:y:05dense_12/dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_12/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_12/dense_12/kernel/Regularizer/mulMul3dense_12/dense_12/kernel/Regularizer/mul/x:output:01dense_12/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_12/dense_12/kernel/Regularizer/addAddV23dense_12/dense_12/kernel/Regularizer/Const:output:0,dense_12/dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
+dense_12/dense_12/kernel/Regularizer/SquareSquareBdense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_12/dense_12/kernel/Regularizer/Sum_1Sum/dense_12/dense_12/kernel/Regularizer/Square:y:05dense_12/dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_12/dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_12/dense_12/kernel/Regularizer/mul_1Mul5dense_12/dense_12/kernel/Regularizer/mul_1/x:output:03dense_12/dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_12/dense_12/kernel/Regularizer/add_1AddV2,dense_12/dense_12/kernel/Regularizer/add:z:0.dense_12/dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp8^dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp;^dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp8^dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp;^dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp8^dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp;^dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2r
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2r
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2r
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
?
?
F__inference_dense_10_layer_call_and_return_conditional_losses_56999650

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999201
input_4;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp?:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp?:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp?:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0}
dense_10/MatMulMatMulinput_4&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
*dense_10/dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(dense_10/dense_10/kernel/Regularizer/AbsAbs?dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_10/dense_10/kernel/Regularizer/SumSum,dense_10/dense_10/kernel/Regularizer/Abs:y:05dense_10/dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_10/dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_10/dense_10/kernel/Regularizer/mulMul3dense_10/dense_10/kernel/Regularizer/mul/x:output:01dense_10/dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_10/dense_10/kernel/Regularizer/addAddV23dense_10/dense_10/kernel/Regularizer/Const:output:0,dense_10/dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+dense_10/dense_10/kernel/Regularizer/SquareSquareBdense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_10/dense_10/kernel/Regularizer/Sum_1Sum/dense_10/dense_10/kernel/Regularizer/Square:y:05dense_10/dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_10/dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_10/dense_10/kernel/Regularizer/mul_1Mul5dense_10/dense_10/kernel/Regularizer/mul_1/x:output:03dense_10/dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_10/dense_10/kernel/Regularizer/add_1AddV2,dense_10/dense_10/kernel/Regularizer/add:z:0.dense_10/dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????do
*dense_11/dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
(dense_11/dense_11/kernel/Regularizer/AbsAbs?dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_11/dense_11/kernel/Regularizer/SumSum,dense_11/dense_11/kernel/Regularizer/Abs:y:05dense_11/dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_11/dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_11/dense_11/kernel/Regularizer/mulMul3dense_11/dense_11/kernel/Regularizer/mul/x:output:01dense_11/dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_11/dense_11/kernel/Regularizer/addAddV23dense_11/dense_11/kernel/Regularizer/Const:output:0,dense_11/dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
+dense_11/dense_11/kernel/Regularizer/SquareSquareBdense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_11/dense_11/kernel/Regularizer/Sum_1Sum/dense_11/dense_11/kernel/Regularizer/Square:y:05dense_11/dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_11/dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_11/dense_11/kernel/Regularizer/mul_1Mul5dense_11/dense_11/kernel/Regularizer/mul_1/x:output:03dense_11/dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_11/dense_11/kernel/Regularizer/add_1AddV2,dense_11/dense_11/kernel/Regularizer/add:z:0.dense_11/dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2o
*dense_12/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
(dense_12/dense_12/kernel/Regularizer/AbsAbs?dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_12/dense_12/kernel/Regularizer/SumSum,dense_12/dense_12/kernel/Regularizer/Abs:y:05dense_12/dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_12/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_12/dense_12/kernel/Regularizer/mulMul3dense_12/dense_12/kernel/Regularizer/mul/x:output:01dense_12/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_12/dense_12/kernel/Regularizer/addAddV23dense_12/dense_12/kernel/Regularizer/Const:output:0,dense_12/dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
+dense_12/dense_12/kernel/Regularizer/SquareSquareBdense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_12/dense_12/kernel/Regularizer/Sum_1Sum/dense_12/dense_12/kernel/Regularizer/Square:y:05dense_12/dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_12/dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_12/dense_12/kernel/Regularizer/mul_1Mul5dense_12/dense_12/kernel/Regularizer/mul_1/x:output:03dense_12/dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_12/dense_12/kernel/Regularizer/add_1AddV2,dense_12/dense_12/kernel/Regularizer/add:z:0.dense_12/dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp8^dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp;^dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp8^dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp;^dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp8^dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp;^dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2r
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2r
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2r
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
?	
?
&__inference_signature_wrapper_56999275
input_4
unknown:
??
	unknown_0:	?
	unknown_1:	?d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_56998037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
?
?
F__inference_dense_12_layer_call_and_return_conditional_losses_56999784

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_56999846J
7dense_11_kernel_regularizer_abs_readvariableop_resource:	?d
identity??.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOpf
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_11_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_11_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_11/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp
??
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999079
input_4;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp?:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp?:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp?:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0}
dense_10/MatMulMatMulinput_4&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
*dense_10/dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(dense_10/dense_10/kernel/Regularizer/AbsAbs?dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_10/dense_10/kernel/Regularizer/SumSum,dense_10/dense_10/kernel/Regularizer/Abs:y:05dense_10/dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_10/dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_10/dense_10/kernel/Regularizer/mulMul3dense_10/dense_10/kernel/Regularizer/mul/x:output:01dense_10/dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_10/dense_10/kernel/Regularizer/addAddV23dense_10/dense_10/kernel/Regularizer/Const:output:0,dense_10/dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+dense_10/dense_10/kernel/Regularizer/SquareSquareBdense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_10/dense_10/kernel/Regularizer/Sum_1Sum/dense_10/dense_10/kernel/Regularizer/Square:y:05dense_10/dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_10/dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_10/dense_10/kernel/Regularizer/mul_1Mul5dense_10/dense_10/kernel/Regularizer/mul_1/x:output:03dense_10/dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_10/dense_10/kernel/Regularizer/add_1AddV2,dense_10/dense_10/kernel/Regularizer/add:z:0.dense_10/dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????do
*dense_11/dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
(dense_11/dense_11/kernel/Regularizer/AbsAbs?dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_11/dense_11/kernel/Regularizer/SumSum,dense_11/dense_11/kernel/Regularizer/Abs:y:05dense_11/dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_11/dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_11/dense_11/kernel/Regularizer/mulMul3dense_11/dense_11/kernel/Regularizer/mul/x:output:01dense_11/dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_11/dense_11/kernel/Regularizer/addAddV23dense_11/dense_11/kernel/Regularizer/Const:output:0,dense_11/dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
+dense_11/dense_11/kernel/Regularizer/SquareSquareBdense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_11/dense_11/kernel/Regularizer/Sum_1Sum/dense_11/dense_11/kernel/Regularizer/Square:y:05dense_11/dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_11/dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_11/dense_11/kernel/Regularizer/mul_1Mul5dense_11/dense_11/kernel/Regularizer/mul_1/x:output:03dense_11/dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_11/dense_11/kernel/Regularizer/add_1AddV2,dense_11/dense_11/kernel/Regularizer/add:z:0.dense_11/dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2o
*dense_12/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
(dense_12/dense_12/kernel/Regularizer/AbsAbs?dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_12/dense_12/kernel/Regularizer/SumSum,dense_12/dense_12/kernel/Regularizer/Abs:y:05dense_12/dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_12/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_12/dense_12/kernel/Regularizer/mulMul3dense_12/dense_12/kernel/Regularizer/mul/x:output:01dense_12/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_12/dense_12/kernel/Regularizer/addAddV23dense_12/dense_12/kernel/Regularizer/Const:output:0,dense_12/dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
+dense_12/dense_12/kernel/Regularizer/SquareSquareBdense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_12/dense_12/kernel/Regularizer/Sum_1Sum/dense_12/dense_12/kernel/Regularizer/Square:y:05dense_12/dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_12/dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_12/dense_12/kernel/Regularizer/mul_1Mul5dense_12/dense_12/kernel/Regularizer/mul_1/x:output:03dense_12/dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_12/dense_12/kernel/Regularizer/add_1AddV2,dense_12/dense_12/kernel/Regularizer/add:z:0.dense_12/dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp8^dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp;^dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp8^dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp;^dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp8^dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp;^dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2r
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2r
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2r
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
?_
?
/__inference_sequential_3_layer_call_fn_56999352

inputs;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_12_layer_call_fn_56999758

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?_
?
/__inference_sequential_3_layer_call_fn_56999429

inputs;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_dense_13_layer_call_and_return_conditional_losses_56999806

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?_
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999583

inputs;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_56999826K
7dense_10_kernel_regularizer_abs_readvariableop_resource:
??
identity??.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOpf
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_10_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_10_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_10/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp
?
?
+__inference_dense_10_layer_call_fn_56999624

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
+__inference_dense_13_layer_call_fn_56999795

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
/__inference_sequential_3_layer_call_fn_56998160
input_4;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp?:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp?:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp?:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0}
dense_10/MatMulMatMulinput_4&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
*dense_10/dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(dense_10/dense_10/kernel/Regularizer/AbsAbs?dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_10/dense_10/kernel/Regularizer/SumSum,dense_10/dense_10/kernel/Regularizer/Abs:y:05dense_10/dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_10/dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_10/dense_10/kernel/Regularizer/mulMul3dense_10/dense_10/kernel/Regularizer/mul/x:output:01dense_10/dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_10/dense_10/kernel/Regularizer/addAddV23dense_10/dense_10/kernel/Regularizer/Const:output:0,dense_10/dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+dense_10/dense_10/kernel/Regularizer/SquareSquareBdense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??}
,dense_10/dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_10/dense_10/kernel/Regularizer/Sum_1Sum/dense_10/dense_10/kernel/Regularizer/Square:y:05dense_10/dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_10/dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_10/dense_10/kernel/Regularizer/mul_1Mul5dense_10/dense_10/kernel/Regularizer/mul_1/x:output:03dense_10/dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_10/dense_10/kernel/Regularizer/add_1AddV2,dense_10/dense_10/kernel/Regularizer/add:z:0.dense_10/dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????do
*dense_11/dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
(dense_11/dense_11/kernel/Regularizer/AbsAbs?dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_11/dense_11/kernel/Regularizer/SumSum,dense_11/dense_11/kernel/Regularizer/Abs:y:05dense_11/dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_11/dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_11/dense_11/kernel/Regularizer/mulMul3dense_11/dense_11/kernel/Regularizer/mul/x:output:01dense_11/dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_11/dense_11/kernel/Regularizer/addAddV23dense_11/dense_11/kernel/Regularizer/Const:output:0,dense_11/dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
+dense_11/dense_11/kernel/Regularizer/SquareSquareBdense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d}
,dense_11/dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_11/dense_11/kernel/Regularizer/Sum_1Sum/dense_11/dense_11/kernel/Regularizer/Square:y:05dense_11/dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_11/dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_11/dense_11/kernel/Regularizer/mul_1Mul5dense_11/dense_11/kernel/Regularizer/mul_1/x:output:03dense_11/dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_11/dense_11/kernel/Regularizer/add_1AddV2,dense_11/dense_11/kernel/Regularizer/add:z:0.dense_11/dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2o
*dense_12/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
(dense_12/dense_12/kernel/Regularizer/AbsAbs?dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_12/dense_12/kernel/Regularizer/SumSum,dense_12/dense_12/kernel/Regularizer/Abs:y:05dense_12/dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: o
*dense_12/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
(dense_12/dense_12/kernel/Regularizer/mulMul3dense_12/dense_12/kernel/Regularizer/mul/x:output:01dense_12/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
(dense_12/dense_12/kernel/Regularizer/addAddV23dense_12/dense_12/kernel/Regularizer/Const:output:0,dense_12/dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
+dense_12/dense_12/kernel/Regularizer/SquareSquareBdense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2}
,dense_12/dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
*dense_12/dense_12/kernel/Regularizer/Sum_1Sum/dense_12/dense_12/kernel/Regularizer/Square:y:05dense_12/dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: q
,dense_12/dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
*dense_12/dense_12/kernel/Regularizer/mul_1Mul5dense_12/dense_12/kernel/Regularizer/mul_1/x:output:03dense_12/dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
*dense_12/dense_12/kernel/Regularizer/add_1AddV2,dense_12/dense_12/kernel/Regularizer/add:z:0.dense_12/dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp8^dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp;^dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp8^dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp;^dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp8^dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp;^dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2r
7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp7dense_10/dense_10/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp:dense_10/dense_10/kernel/Regularizer/Square/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2r
7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp7dense_11/dense_11/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp:dense_11/dense_11/kernel/Regularizer/Square/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2r
7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp7dense_12/dense_12/kernel/Regularizer/Abs/ReadVariableOp2x
:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp:dense_12/dense_12/kernel/Regularizer/Square/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_4
?_
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999506

inputs;
'dense_10_matmul_readvariableop_resource:
??7
(dense_10_biasadd_readvariableop_resource:	?:
'dense_11_matmul_readvariableop_resource:	?d6
(dense_11_biasadd_readvariableop_resource:d9
'dense_12_matmul_readvariableop_resource:d26
(dense_12_biasadd_readvariableop_resource:29
'dense_13_matmul_readvariableop_resource:26
(dense_13_biasadd_readvariableop_resource:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?.dense_10/kernel/Regularizer/Abs/ReadVariableOp?1dense_10/kernel/Regularizer/Square/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_10/kernel/Regularizer/AbsAbs6dense_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_10/kernel/Regularizer/SumSum#dense_10/kernel/Regularizer/Abs:y:0,dense_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/Const:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??t
#dense_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_10/kernel/Regularizer/Sum_1Sum&dense_10/kernel/Regularizer/Square:y:0,dense_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_10/kernel/Regularizer/mul_1Mul,dense_10/kernel/Regularizer/mul_1/x:output:0*dense_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_10/kernel/Regularizer/add_1AddV2#dense_10/kernel/Regularizer/add:z:0%dense_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp/^dense_10/kernel/Regularizer/Abs/ReadVariableOp2^dense_10/kernel/Regularizer/Square/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2`
.dense_10/kernel/Regularizer/Abs/ReadVariableOp.dense_10/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_10/kernel/Regularizer/Square/ReadVariableOp1dense_10/kernel/Regularizer/Square/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_11_layer_call_fn_56999691

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????df
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_dense_11_layer_call_and_return_conditional_losses_56999717

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_11/kernel/Regularizer/Abs/ReadVariableOp?1dense_11/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????df
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_11/kernel/Regularizer/AbsAbs6dense_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_11/kernel/Regularizer/SumSum#dense_11/kernel/Regularizer/Abs:y:0,dense_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/Const:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?dt
#dense_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_11/kernel/Regularizer/Sum_1Sum&dense_11/kernel/Regularizer/Square:y:0,dense_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_11/kernel/Regularizer/mul_1Mul,dense_11/kernel/Regularizer/mul_1/x:output:0*dense_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_11/kernel/Regularizer/add_1AddV2#dense_11/kernel/Regularizer/add:z:0%dense_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_11/kernel/Regularizer/Abs/ReadVariableOp2^dense_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_11/kernel/Regularizer/Abs/ReadVariableOp.dense_11/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_11/kernel/Regularizer/Square/ReadVariableOp1dense_11/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_56999866I
7dense_12_kernel_regularizer_abs_readvariableop_resource:d2
identity??.dense_12/kernel/Regularizer/Abs/ReadVariableOp?1dense_12/kernel/Regularizer/Square/ReadVariableOpf
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_12_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_12/kernel/Regularizer/AbsAbs6dense_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_12/kernel/Regularizer/SumSum#dense_12/kernel/Regularizer/Abs:y:0,dense_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_12/kernel/Regularizer/addAddV2*dense_12/kernel/Regularizer/Const:output:0#dense_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_12_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d2*
dtype0?
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2t
#dense_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_12/kernel/Regularizer/Sum_1Sum&dense_12/kernel/Regularizer/Square:y:0,dense_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
!dense_12/kernel/Regularizer/mul_1Mul,dense_12/kernel/Regularizer/mul_1/x:output:0*dense_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_12/kernel/Regularizer/add_1AddV2#dense_12/kernel/Regularizer/add:z:0%dense_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_12/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^dense_12/kernel/Regularizer/Abs/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_12/kernel/Regularizer/Abs/ReadVariableOp.dense_12/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp"?-
saver_filename:0
Identity:0Identity_348"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_41
serving_default_input_4:0??????????<
dense_130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?d
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
\__call__
*]&call_and_return_all_conditional_losses
^_default_save_signature"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemLmMmNmOmPmQmRmSvTvUvVvWvXvYvZv["
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
5
g0
h1
i2"
trackable_list_wrapper
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
\__call__
^_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
#:!
??2dense_10/kernel
:?2dense_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
": 	?d2dense_11/kernel
:d2dense_11/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
!:d22dense_12/kernel
:22dense_12/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:22dense_13/kernel
:2dense_13/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metric
^
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
(:&
??2Adam/dense_10/kernel/m
!:?2Adam/dense_10/bias/m
':%	?d2Adam/dense_11/kernel/m
 :d2Adam/dense_11/bias/m
&:$d22Adam/dense_12/kernel/m
 :22Adam/dense_12/bias/m
&:$22Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
(:&
??2Adam/dense_10/kernel/v
!:?2Adam/dense_10/bias/v
':%	?d2Adam/dense_11/kernel/v
 :d2Adam/dense_11/bias/v
&:$d22Adam/dense_12/kernel/v
 :22Adam/dense_12/bias/v
&:$22Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
?2?
/__inference_sequential_3_layer_call_fn_56998160
/__inference_sequential_3_layer_call_fn_56999352
/__inference_sequential_3_layer_call_fn_56999429
/__inference_sequential_3_layer_call_fn_56998957?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999506
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999583
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999079
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999201?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_56998037input_4"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_10_layer_call_fn_56999624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_10_layer_call_and_return_conditional_losses_56999650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_11_layer_call_fn_56999691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_11_layer_call_and_return_conditional_losses_56999717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_12_layer_call_fn_56999758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_12_layer_call_and_return_conditional_losses_56999784?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_13_layer_call_fn_56999795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_13_layer_call_and_return_conditional_losses_56999806?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_56999826?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_56999846?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_56999866?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
&__inference_signature_wrapper_56999275input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_56998037r1?.
'?$
"?
input_4??????????
? "3?0
.
dense_13"?
dense_13??????????
F__inference_dense_10_layer_call_and_return_conditional_losses_56999650^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_10_layer_call_fn_56999624Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_11_layer_call_and_return_conditional_losses_56999717]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? 
+__inference_dense_11_layer_call_fn_56999691P0?-
&?#
!?
inputs??????????
? "??????????d?
F__inference_dense_12_layer_call_and_return_conditional_losses_56999784\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????2
? ~
+__inference_dense_12_layer_call_fn_56999758O/?,
%?"
 ?
inputs?????????d
? "??????????2?
F__inference_dense_13_layer_call_and_return_conditional_losses_56999806\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? ~
+__inference_dense_13_layer_call_fn_56999795O/?,
%?"
 ?
inputs?????????2
? "??????????=
__inference_loss_fn_0_56999826?

? 
? "? =
__inference_loss_fn_1_56999846?

? 
? "? =
__inference_loss_fn_2_56999866?

? 
? "? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999079l9?6
/?,
"?
input_4??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999201l9?6
/?,
"?
input_4??????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999506k8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_56999583k8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_3_layer_call_fn_56998160_9?6
/?,
"?
input_4??????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_56998957_9?6
/?,
"?
input_4??????????
p

 
? "???????????
/__inference_sequential_3_layer_call_fn_56999352^8?5
.?+
!?
inputs??????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_56999429^8?5
.?+
!?
inputs??????????
p

 
? "???????????
&__inference_signature_wrapper_56999275}<?9
? 
2?/
-
input_4"?
input_4??????????"3?0
.
dense_13"?
dense_13?????????