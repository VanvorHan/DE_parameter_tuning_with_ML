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
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:d*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:d2*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:2*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:2*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
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
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	?d*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:d2*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:2*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	?d*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:d2*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:2*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:2*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
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
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
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
&__inference_signature_wrapper_15307190
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

SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst"/device:CPU:0*0
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
Z
AssignVariableOpAssignVariableOpdense/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_1AssignVariableOp
dense/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_2AssignVariableOpdense_1/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_3AssignVariableOpdense_1/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_4AssignVariableOpdense_2/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_5AssignVariableOpdense_2/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_6AssignVariableOpdense_3/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_7AssignVariableOpdense_3/bias
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
e
AssignVariableOp_17AssignVariableOpAdam/dense/kernel/mIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_18AssignVariableOpAdam/dense/bias/mIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_19AssignVariableOpAdam/dense_1/kernel/mIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_20AssignVariableOpAdam/dense_1/bias/mIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_21AssignVariableOpAdam/dense_2/kernel/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_22AssignVariableOpAdam/dense_2/bias/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_23AssignVariableOpAdam/dense_3/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_24AssignVariableOpAdam/dense_3/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_25AssignVariableOpAdam/dense/kernel/vIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_26AssignVariableOpAdam/dense/bias/vIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_27AssignVariableOpAdam/dense_1/kernel/vIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_28AssignVariableOpAdam/dense_1/bias/vIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_29AssignVariableOpAdam/dense_2/kernel/vIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_30AssignVariableOpAdam/dense_2/bias/vIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_31AssignVariableOpAdam/dense_3/kernel/vIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_32AssignVariableOpAdam/dense_3/bias/vIdentity_33"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_34Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??

?\
?
-__inference_sequential_layer_call_fn_15307344

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_15307761I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	?d
identity??-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOpe
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
?*
?
#__inference__wrapped_model_15305952
input_1C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?D
1sequential_dense_1_matmul_readvariableop_resource:	?d@
2sequential_dense_1_biasadd_readvariableop_resource:dC
1sequential_dense_2_matmul_readvariableop_resource:d2@
2sequential_dense_2_biasadd_readvariableop_resource:2C
1sequential_dense_3_matmul_readvariableop_resource:2@
2sequential_dense_3_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/dense/MatMulMatMulinput_1.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dv
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2v
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
sequential/dense_3/MatMulMatMul%sequential/dense_2/Relu:activations:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_3/SoftmaxSoftmax#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????s
IdentityIdentity$sequential/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?
-__inference_sequential_layer_call_fn_15306075
input_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?1dense/dense/kernel/Regularizer/Abs/ReadVariableOp?4dense/dense/kernel/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp?8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp?8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
$dense/dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1dense/dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense/dense/kernel/Regularizer/AbsAbs9dense/dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
"dense/dense/kernel/Regularizer/SumSum&dense/dense/kernel/Regularizer/Abs:y:0/dense/dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$dense/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
"dense/dense/kernel/Regularizer/mulMul-dense/dense/kernel/Regularizer/mul/x:output:0+dense/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
"dense/dense/kernel/Regularizer/addAddV2-dense/dense/kernel/Regularizer/Const:output:0&dense/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
4dense/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%dense/dense/kernel/Regularizer/SquareSquare<dense/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
$dense/dense/kernel/Regularizer/Sum_1Sum)dense/dense/kernel/Regularizer/Square:y:0/dense/dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&dense/dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$dense/dense/kernel/Regularizer/mul_1Mul/dense/dense/kernel/Regularizer/mul_1/x:output:0-dense/dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
$dense/dense/kernel/Regularizer/add_1AddV2&dense/dense/kernel/Regularizer/add:z:0(dense/dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????dm
(dense_1/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
&dense_1/dense_1/kernel/Regularizer/AbsAbs=dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_1/dense_1/kernel/Regularizer/SumSum*dense_1/dense_1/kernel/Regularizer/Abs:y:03dense_1/dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_1/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_1/dense_1/kernel/Regularizer/mulMul1dense_1/dense_1/kernel/Regularizer/mul/x:output:0/dense_1/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_1/dense_1/kernel/Regularizer/addAddV21dense_1/dense_1/kernel/Regularizer/Const:output:0*dense_1/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
)dense_1/dense_1/kernel/Regularizer/SquareSquare@dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_1/dense_1/kernel/Regularizer/Sum_1Sum-dense_1/dense_1/kernel/Regularizer/Square:y:03dense_1/dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_1/dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_1/dense_1/kernel/Regularizer/mul_1Mul3dense_1/dense_1/kernel/Regularizer/mul_1/x:output:01dense_1/dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_1/dense_1/kernel/Regularizer/add_1AddV2*dense_1/dense_1/kernel/Regularizer/add:z:0,dense_1/dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
(dense_2/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
&dense_2/dense_2/kernel/Regularizer/AbsAbs=dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_2/dense_2/kernel/Regularizer/SumSum*dense_2/dense_2/kernel/Regularizer/Abs:y:03dense_2/dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_2/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_2/dense_2/kernel/Regularizer/mulMul1dense_2/dense_2/kernel/Regularizer/mul/x:output:0/dense_2/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_2/dense_2/kernel/Regularizer/addAddV21dense_2/dense_2/kernel/Regularizer/Const:output:0*dense_2/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
)dense_2/dense_2/kernel/Regularizer/SquareSquare@dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_2/dense_2/kernel/Regularizer/Sum_1Sum-dense_2/dense_2/kernel/Regularizer/Square:y:03dense_2/dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_2/dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_2/dense_2/kernel/Regularizer/mul_1Mul3dense_2/dense_2/kernel/Regularizer/mul_1/x:output:01dense_2/dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_2/dense_2/kernel/Regularizer/add_1AddV2*dense_2/dense_2/kernel/Regularizer/add:z:0,dense_2/dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp2^dense/dense/kernel/Regularizer/Abs/ReadVariableOp5^dense/dense/kernel/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp6^dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp9^dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp6^dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp9^dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2f
1dense/dense/kernel/Regularizer/Abs/ReadVariableOp1dense/dense/kernel/Regularizer/Abs/ReadVariableOp2l
4dense/dense/kernel/Regularizer/Square/ReadVariableOp4dense/dense/kernel/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2n
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2n
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?
-__inference_sequential_layer_call_fn_15306872
input_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?1dense/dense/kernel/Regularizer/Abs/ReadVariableOp?4dense/dense/kernel/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp?8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp?8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
$dense/dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1dense/dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense/dense/kernel/Regularizer/AbsAbs9dense/dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
"dense/dense/kernel/Regularizer/SumSum&dense/dense/kernel/Regularizer/Abs:y:0/dense/dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$dense/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
"dense/dense/kernel/Regularizer/mulMul-dense/dense/kernel/Regularizer/mul/x:output:0+dense/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
"dense/dense/kernel/Regularizer/addAddV2-dense/dense/kernel/Regularizer/Const:output:0&dense/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
4dense/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%dense/dense/kernel/Regularizer/SquareSquare<dense/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
$dense/dense/kernel/Regularizer/Sum_1Sum)dense/dense/kernel/Regularizer/Square:y:0/dense/dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&dense/dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$dense/dense/kernel/Regularizer/mul_1Mul/dense/dense/kernel/Regularizer/mul_1/x:output:0-dense/dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
$dense/dense/kernel/Regularizer/add_1AddV2&dense/dense/kernel/Regularizer/add:z:0(dense/dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????dm
(dense_1/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
&dense_1/dense_1/kernel/Regularizer/AbsAbs=dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_1/dense_1/kernel/Regularizer/SumSum*dense_1/dense_1/kernel/Regularizer/Abs:y:03dense_1/dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_1/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_1/dense_1/kernel/Regularizer/mulMul1dense_1/dense_1/kernel/Regularizer/mul/x:output:0/dense_1/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_1/dense_1/kernel/Regularizer/addAddV21dense_1/dense_1/kernel/Regularizer/Const:output:0*dense_1/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
)dense_1/dense_1/kernel/Regularizer/SquareSquare@dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_1/dense_1/kernel/Regularizer/Sum_1Sum-dense_1/dense_1/kernel/Regularizer/Square:y:03dense_1/dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_1/dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_1/dense_1/kernel/Regularizer/mul_1Mul3dense_1/dense_1/kernel/Regularizer/mul_1/x:output:01dense_1/dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_1/dense_1/kernel/Regularizer/add_1AddV2*dense_1/dense_1/kernel/Regularizer/add:z:0,dense_1/dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
(dense_2/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
&dense_2/dense_2/kernel/Regularizer/AbsAbs=dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_2/dense_2/kernel/Regularizer/SumSum*dense_2/dense_2/kernel/Regularizer/Abs:y:03dense_2/dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_2/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_2/dense_2/kernel/Regularizer/mulMul1dense_2/dense_2/kernel/Regularizer/mul/x:output:0/dense_2/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_2/dense_2/kernel/Regularizer/addAddV21dense_2/dense_2/kernel/Regularizer/Const:output:0*dense_2/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
)dense_2/dense_2/kernel/Regularizer/SquareSquare@dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_2/dense_2/kernel/Regularizer/Sum_1Sum-dense_2/dense_2/kernel/Regularizer/Square:y:03dense_2/dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_2/dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_2/dense_2/kernel/Regularizer/mul_1Mul3dense_2/dense_2/kernel/Regularizer/mul_1/x:output:01dense_2/dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_2/dense_2/kernel/Regularizer/add_1AddV2*dense_2/dense_2/kernel/Regularizer/add:z:0,dense_2/dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp2^dense/dense/kernel/Regularizer/Abs/ReadVariableOp5^dense/dense/kernel/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp6^dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp9^dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp6^dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp9^dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2f
1dense/dense/kernel/Regularizer/Abs/ReadVariableOp1dense/dense/kernel/Regularizer/Abs/ReadVariableOp2l
4dense/dense/kernel/Regularizer/Square/ReadVariableOp4dense/dense/kernel/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2n
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2n
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?\
?
H__inference_sequential_layer_call_and_return_conditional_losses_15307421

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_15307606

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
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
:?????????de
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_15307632

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
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
:?????????de
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_15307539

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
H__inference_sequential_layer_call_and_return_conditional_losses_15307498

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
Ü
?
H__inference_sequential_layer_call_and_return_conditional_losses_15307116
input_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?1dense/dense/kernel/Regularizer/Abs/ReadVariableOp?4dense/dense/kernel/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp?8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp?8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
$dense/dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1dense/dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense/dense/kernel/Regularizer/AbsAbs9dense/dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
"dense/dense/kernel/Regularizer/SumSum&dense/dense/kernel/Regularizer/Abs:y:0/dense/dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$dense/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
"dense/dense/kernel/Regularizer/mulMul-dense/dense/kernel/Regularizer/mul/x:output:0+dense/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
"dense/dense/kernel/Regularizer/addAddV2-dense/dense/kernel/Regularizer/Const:output:0&dense/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
4dense/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%dense/dense/kernel/Regularizer/SquareSquare<dense/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
$dense/dense/kernel/Regularizer/Sum_1Sum)dense/dense/kernel/Regularizer/Square:y:0/dense/dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&dense/dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$dense/dense/kernel/Regularizer/mul_1Mul/dense/dense/kernel/Regularizer/mul_1/x:output:0-dense/dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
$dense/dense/kernel/Regularizer/add_1AddV2&dense/dense/kernel/Regularizer/add:z:0(dense/dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????dm
(dense_1/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
&dense_1/dense_1/kernel/Regularizer/AbsAbs=dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_1/dense_1/kernel/Regularizer/SumSum*dense_1/dense_1/kernel/Regularizer/Abs:y:03dense_1/dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_1/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_1/dense_1/kernel/Regularizer/mulMul1dense_1/dense_1/kernel/Regularizer/mul/x:output:0/dense_1/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_1/dense_1/kernel/Regularizer/addAddV21dense_1/dense_1/kernel/Regularizer/Const:output:0*dense_1/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
)dense_1/dense_1/kernel/Regularizer/SquareSquare@dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_1/dense_1/kernel/Regularizer/Sum_1Sum-dense_1/dense_1/kernel/Regularizer/Square:y:03dense_1/dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_1/dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_1/dense_1/kernel/Regularizer/mul_1Mul3dense_1/dense_1/kernel/Regularizer/mul_1/x:output:01dense_1/dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_1/dense_1/kernel/Regularizer/add_1AddV2*dense_1/dense_1/kernel/Regularizer/add:z:0,dense_1/dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
(dense_2/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
&dense_2/dense_2/kernel/Regularizer/AbsAbs=dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_2/dense_2/kernel/Regularizer/SumSum*dense_2/dense_2/kernel/Regularizer/Abs:y:03dense_2/dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_2/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_2/dense_2/kernel/Regularizer/mulMul1dense_2/dense_2/kernel/Regularizer/mul/x:output:0/dense_2/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_2/dense_2/kernel/Regularizer/addAddV21dense_2/dense_2/kernel/Regularizer/Const:output:0*dense_2/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
)dense_2/dense_2/kernel/Regularizer/SquareSquare@dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_2/dense_2/kernel/Regularizer/Sum_1Sum-dense_2/dense_2/kernel/Regularizer/Square:y:03dense_2/dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_2/dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_2/dense_2/kernel/Regularizer/mul_1Mul3dense_2/dense_2/kernel/Regularizer/mul_1/x:output:01dense_2/dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_2/dense_2/kernel/Regularizer/add_1AddV2*dense_2/dense_2/kernel/Regularizer/add:z:0,dense_2/dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp2^dense/dense/kernel/Regularizer/Abs/ReadVariableOp5^dense/dense/kernel/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp6^dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp9^dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp6^dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp9^dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2f
1dense/dense/kernel/Regularizer/Abs/ReadVariableOp1dense/dense/kernel/Regularizer/Abs/ReadVariableOp2l
4dense/dense/kernel/Regularizer/Square/ReadVariableOp4dense/dense/kernel/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2n
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2n
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
&__inference_signature_wrapper_15307190
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_15305952o
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
_user_specified_name	input_1
?
?
C__inference_dense_layer_call_and_return_conditional_losses_15307565

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
-__inference_sequential_layer_call_fn_15307267

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_2_layer_call_fn_15307673

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOpt
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
:?????????2e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
E__inference_dense_2_layer_call_and_return_conditional_losses_15307699

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOpt
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
:?????????2e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
*__inference_dense_3_layer_call_fn_15307710

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
?
?
__inference_loss_fn_2_15307781H
6dense_2_kernel_regularizer_abs_readvariableop_resource:d2
identity??-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOpe
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_2/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
Ü
?
H__inference_sequential_layer_call_and_return_conditional_losses_15306994
input_18
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:d25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?1dense/dense/kernel/Regularizer/Abs/ReadVariableOp?4dense/dense/kernel/Regularizer/Square/ReadVariableOp?+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp?8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp?-dense_1/kernel/Regularizer/Abs/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp?8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
$dense/dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1dense/dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
"dense/dense/kernel/Regularizer/AbsAbs9dense/dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
"dense/dense/kernel/Regularizer/SumSum&dense/dense/kernel/Regularizer/Abs:y:0/dense/dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: i
$dense/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
"dense/dense/kernel/Regularizer/mulMul-dense/dense/kernel/Regularizer/mul/x:output:0+dense/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
"dense/dense/kernel/Regularizer/addAddV2-dense/dense/kernel/Regularizer/Const:output:0&dense/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
4dense/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%dense/dense/kernel/Regularizer/SquareSquare<dense/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??w
&dense/dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
$dense/dense/kernel/Regularizer/Sum_1Sum)dense/dense/kernel/Regularizer/Square:y:0/dense/dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: k
&dense/dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
$dense/dense/kernel/Regularizer/mul_1Mul/dense/dense/kernel/Regularizer/mul_1/x:output:0-dense/dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
$dense/dense/kernel/Regularizer/add_1AddV2&dense/dense/kernel/Regularizer/add:z:0(dense/dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????dm
(dense_1/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
&dense_1/dense_1/kernel/Regularizer/AbsAbs=dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_1/dense_1/kernel/Regularizer/SumSum*dense_1/dense_1/kernel/Regularizer/Abs:y:03dense_1/dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_1/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_1/dense_1/kernel/Regularizer/mulMul1dense_1/dense_1/kernel/Regularizer/mul/x:output:0/dense_1/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_1/dense_1/kernel/Regularizer/addAddV21dense_1/dense_1/kernel/Regularizer/Const:output:0*dense_1/dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
)dense_1/dense_1/kernel/Regularizer/SquareSquare@dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?d{
*dense_1/dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_1/dense_1/kernel/Regularizer/Sum_1Sum-dense_1/dense_1/kernel/Regularizer/Square:y:03dense_1/dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_1/dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_1/dense_1/kernel/Regularizer/mul_1Mul3dense_1/dense_1/kernel/Regularizer/mul_1/x:output:01dense_1/dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_1/dense_1/kernel/Regularizer/add_1AddV2*dense_1/dense_1/kernel/Regularizer/add:z:0,dense_1/dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
(dense_2/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
&dense_2/dense_2/kernel/Regularizer/AbsAbs=dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
&dense_2/dense_2/kernel/Regularizer/SumSum*dense_2/dense_2/kernel/Regularizer/Abs:y:03dense_2/dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: m
(dense_2/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&dense_2/dense_2/kernel/Regularizer/mulMul1dense_2/dense_2/kernel/Regularizer/mul/x:output:0/dense_2/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
&dense_2/dense_2/kernel/Regularizer/addAddV21dense_2/dense_2/kernel/Regularizer/Const:output:0*dense_2/dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
)dense_2/dense_2/kernel/Regularizer/SquareSquare@dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2{
*dense_2/dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
(dense_2/dense_2/kernel/Regularizer/Sum_1Sum-dense_2/dense_2/kernel/Regularizer/Square:y:03dense_2/dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: o
*dense_2/dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(dense_2/dense_2/kernel/Regularizer/mul_1Mul3dense_2/dense_2/kernel/Regularizer/mul_1/x:output:01dense_2/dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
(dense_2/dense_2/kernel/Regularizer/add_1AddV2*dense_2/dense_2/kernel/Regularizer/add:z:0,dense_2/dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?ds
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2s
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp2^dense/dense/kernel/Regularizer/Abs/ReadVariableOp5^dense/dense/kernel/Regularizer/Square/ReadVariableOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp6^dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp9^dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp6^dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp9^dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2f
1dense/dense/kernel/Regularizer/Abs/ReadVariableOp1dense/dense/kernel/Regularizer/Abs/ReadVariableOp2l
4dense/dense/kernel/Regularizer/Square/ReadVariableOp4dense/dense/kernel/Regularizer/Square/ReadVariableOp2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2n
5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp5dense_1/dense_1/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp8dense_1/dense_1/kernel/Regularizer/Square/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2n
5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp5dense_2/dense_2/kernel/Regularizer/Abs/ReadVariableOp2t
8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp8dense_2/dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
E__inference_dense_3_layer_call_and_return_conditional_losses_15307721

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
?
?
__inference_loss_fn_0_15307741H
4dense_kernel_regularizer_abs_readvariableop_resource:
??
identity??+dense/kernel/Regularizer/Abs/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOpc
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/Const:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
 dense/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: e
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: `
IdentityIdentity"dense/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp,^dense/kernel/Regularizer/Abs/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+dense/kernel/Regularizer/Abs/ReadVariableOp+dense/kernel/Regularizer/Abs/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp"?-
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
input_11
serving_default_input_1:0??????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?c
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
 :
??2dense/kernel
:?2
dense/bias
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
!:	?d2dense_1/kernel
:d2dense_1/bias
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
 :d22dense_2/kernel
:22dense_2/bias
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
 :22dense_3/kernel
:2dense_3/bias
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
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
&:$	?d2Adam/dense_1/kernel/m
:d2Adam/dense_1/bias/m
%:#d22Adam/dense_2/kernel/m
:22Adam/dense_2/bias/m
%:#22Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
&:$	?d2Adam/dense_1/kernel/v
:d2Adam/dense_1/bias/v
%:#d22Adam/dense_2/kernel/v
:22Adam/dense_2/bias/v
%:#22Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
?2?
-__inference_sequential_layer_call_fn_15306075
-__inference_sequential_layer_call_fn_15307267
-__inference_sequential_layer_call_fn_15307344
-__inference_sequential_layer_call_fn_15306872?
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
H__inference_sequential_layer_call_and_return_conditional_losses_15307421
H__inference_sequential_layer_call_and_return_conditional_losses_15307498
H__inference_sequential_layer_call_and_return_conditional_losses_15306994
H__inference_sequential_layer_call_and_return_conditional_losses_15307116?
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
#__inference__wrapped_model_15305952input_1"?
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
(__inference_dense_layer_call_fn_15307539?
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
C__inference_dense_layer_call_and_return_conditional_losses_15307565?
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
*__inference_dense_1_layer_call_fn_15307606?
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
E__inference_dense_1_layer_call_and_return_conditional_losses_15307632?
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
*__inference_dense_2_layer_call_fn_15307673?
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
E__inference_dense_2_layer_call_and_return_conditional_losses_15307699?
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
*__inference_dense_3_layer_call_fn_15307710?
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
E__inference_dense_3_layer_call_and_return_conditional_losses_15307721?
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
__inference_loss_fn_0_15307741?
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
__inference_loss_fn_1_15307761?
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
__inference_loss_fn_2_15307781?
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
&__inference_signature_wrapper_15307190input_1"?
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
#__inference__wrapped_model_15305952p1?.
'?$
"?
input_1??????????
? "1?.
,
dense_3!?
dense_3??????????
E__inference_dense_1_layer_call_and_return_conditional_losses_15307632]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ~
*__inference_dense_1_layer_call_fn_15307606P0?-
&?#
!?
inputs??????????
? "??????????d?
E__inference_dense_2_layer_call_and_return_conditional_losses_15307699\/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????2
? }
*__inference_dense_2_layer_call_fn_15307673O/?,
%?"
 ?
inputs?????????d
? "??????????2?
E__inference_dense_3_layer_call_and_return_conditional_losses_15307721\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? }
*__inference_dense_3_layer_call_fn_15307710O/?,
%?"
 ?
inputs?????????2
? "???????????
C__inference_dense_layer_call_and_return_conditional_losses_15307565^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_layer_call_fn_15307539Q0?-
&?#
!?
inputs??????????
? "???????????=
__inference_loss_fn_0_15307741?

? 
? "? =
__inference_loss_fn_1_15307761?

? 
? "? =
__inference_loss_fn_2_15307781?

? 
? "? ?
H__inference_sequential_layer_call_and_return_conditional_losses_15306994l9?6
/?,
"?
input_1??????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_15307116l9?6
/?,
"?
input_1??????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_15307421k8?5
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
H__inference_sequential_layer_call_and_return_conditional_losses_15307498k8?5
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
-__inference_sequential_layer_call_fn_15306075_9?6
/?,
"?
input_1??????????
p 

 
? "???????????
-__inference_sequential_layer_call_fn_15306872_9?6
/?,
"?
input_1??????????
p

 
? "???????????
-__inference_sequential_layer_call_fn_15307267^8?5
.?+
!?
inputs??????????
p 

 
? "???????????
-__inference_sequential_layer_call_fn_15307344^8?5
.?+
!?
inputs??????????
p

 
? "???????????
&__inference_signature_wrapper_15307190{<?9
? 
2?/
-
input_1"?
input_1??????????"1?.
,
dense_3!?
dense_3?????????