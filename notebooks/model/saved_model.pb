??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.5.02unknown8??
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
%simple_rnn_2/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%simple_rnn_2/simple_rnn_cell_2/kernel
?
9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_2/simple_rnn_cell_2/kernel*
_output_shapes
:	?*
dtype0
?
/simple_rnn_2/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*@
shared_name1/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
?
Csimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
#simple_rnn_2/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#simple_rnn_2/simple_rnn_cell_2/bias
?
7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_2/simple_rnn_cell_2/bias*
_output_shapes	
:?*
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
?
RMSprop/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameRMSprop/dense_3/kernel/rms
?
.RMSprop/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/rms*
_output_shapes
:	?*
dtype0
?
RMSprop/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_3/bias/rms
?
,RMSprop/dense_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/rms*
_output_shapes
:*
dtype0
?
1RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*B
shared_name31RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms
?
ERMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms/Read/ReadVariableOpReadVariableOp1RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms*
_output_shapes
:	?*
dtype0
?
;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*L
shared_name=;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms
?
ORMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms* 
_output_shapes
:
??*
dtype0
?
/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms
?
CRMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms/Read/ReadVariableOpReadVariableOp/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
l
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
v
iter
	decay
learning_rate
momentum
rho	rms;	rms<	rms=	rms>	rms?
#
0
1
2
3
4
 
#
0
1
2
3
4
?
non_trainable_variables
metrics
	variables
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables

!layers
 
~

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
 

0
1
2
 

0
1
2
?
&non_trainable_variables
'metrics
	variables
(layer_metrics

)states
*layer_regularization_losses
regularization_losses
trainable_variables

+layers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
,non_trainable_variables
-metrics
	variables
.layer_metrics
/layer_regularization_losses
regularization_losses
trainable_variables

0layers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_2/simple_rnn_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_2/simple_rnn_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

10
 
 

0
1

0
1
2
 

0
1
2
?
2non_trainable_variables
3metrics
"	variables
4layer_metrics
5layer_regularization_losses
#regularization_losses
$trainable_variables

6layers
 
 
 
 
 

	0
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
??
VARIABLE_VALUERMSprop/dense_3/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_3/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_simple_rnn_2_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_2_input%simple_rnn_2/simple_rnn_cell_2/kernel#simple_rnn_2/simple_rnn_cell_2/bias/simple_rnn_2/simple_rnn_cell_2/recurrent_kerneldense_3/kerneldense_3/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_21946
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpCsimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOp7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.RMSprop/dense_3/kernel/rms/Read/ReadVariableOp,RMSprop/dense_3/bias/rms/Read/ReadVariableOpERMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms/Read/ReadVariableOpORMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms/Read/ReadVariableOpCRMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_22859
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rho%simple_rnn_2/simple_rnn_cell_2/kernel/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel#simple_rnn_2/simple_rnn_cell_2/biastotalcountRMSprop/dense_3/kernel/rmsRMSprop/dense_3/bias/rms1RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_22920??
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21648

inputsC
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21582*
condR
while_cond_21581*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_2_layer_call_fn_22671
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_211242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_22714

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
while_body_22370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_21582
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_21168

inputs

states1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
,__inference_sequential_2_layer_call_fn_22197

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_216732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_22257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22257___redundant_placeholder03
/while_while_cond_22257___redundant_placeholder13
/while_while_cond_22257___redundant_placeholder23
/while_while_cond_22257___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_simple_rnn_2_layer_call_fn_22704

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_218212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_21686
simple_rnn_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_216732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?R
?
*sequential_2_simple_rnn_2_while_body_20924P
Lsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_loop_counterV
Rsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_maximum_iterations/
+sequential_2_simple_rnn_2_while_placeholder1
-sequential_2_simple_rnn_2_while_placeholder_11
-sequential_2_simple_rnn_2_while_placeholder_2O
Ksequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_strided_slice_1_0?
?sequential_2_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0e
Rsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?b
Ssequential_2_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?h
Tsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??,
(sequential_2_simple_rnn_2_while_identity.
*sequential_2_simple_rnn_2_while_identity_1.
*sequential_2_simple_rnn_2_while_identity_2.
*sequential_2_simple_rnn_2_while_identity_3.
*sequential_2_simple_rnn_2_while_identity_4M
Isequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_strided_slice_1?
?sequential_2_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorc
Psequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:	?`
Qsequential_2_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?f
Rsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????Hsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Gsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Isequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Qsequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2S
Qsequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Csequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_2_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0+sequential_2_simple_rnn_2_while_placeholderZsequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02E
Csequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
Gsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpRsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02I
Gsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
8sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMulJsequential_2/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul?
Hsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpSsequential_2_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02J
Hsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
9sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAddBsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Psequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd?
Isequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpTsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02K
Isequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
:sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul-sequential_2_simple_rnn_2_while_placeholder_2Qsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2<
:sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1?
5sequential_2/simple_rnn_2/while/simple_rnn_cell_2/addAddV2Bsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:0Dsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????27
5sequential_2/simple_rnn_2/while/simple_rnn_cell_2/add?
6sequential_2/simple_rnn_2/while/simple_rnn_cell_2/TanhTanh9sequential_2/simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????28
6sequential_2/simple_rnn_2/while/simple_rnn_cell_2/Tanh?
Dsequential_2/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_2_simple_rnn_2_while_placeholder_1+sequential_2_simple_rnn_2_while_placeholder:sequential_2/simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_2/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem?
%sequential_2/simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_2/simple_rnn_2/while/add/y?
#sequential_2/simple_rnn_2/while/addAddV2+sequential_2_simple_rnn_2_while_placeholder.sequential_2/simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_2/simple_rnn_2/while/add?
'sequential_2/simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_2/simple_rnn_2/while/add_1/y?
%sequential_2/simple_rnn_2/while/add_1AddV2Lsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_loop_counter0sequential_2/simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_2/simple_rnn_2/while/add_1?
(sequential_2/simple_rnn_2/while/IdentityIdentity)sequential_2/simple_rnn_2/while/add_1:z:0I^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpH^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpJ^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_2/simple_rnn_2/while/Identity?
*sequential_2/simple_rnn_2/while/Identity_1IdentityRsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_maximum_iterationsI^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpH^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpJ^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_2/simple_rnn_2/while/Identity_1?
*sequential_2/simple_rnn_2/while/Identity_2Identity'sequential_2/simple_rnn_2/while/add:z:0I^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpH^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpJ^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_2/simple_rnn_2/while/Identity_2?
*sequential_2/simple_rnn_2/while/Identity_3IdentityTsequential_2/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpH^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpJ^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_2/simple_rnn_2/while/Identity_3?
*sequential_2/simple_rnn_2/while/Identity_4Identity:sequential_2/simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0I^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpH^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpJ^sequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2,
*sequential_2/simple_rnn_2/while/Identity_4"]
(sequential_2_simple_rnn_2_while_identity1sequential_2/simple_rnn_2/while/Identity:output:0"a
*sequential_2_simple_rnn_2_while_identity_13sequential_2/simple_rnn_2/while/Identity_1:output:0"a
*sequential_2_simple_rnn_2_while_identity_23sequential_2/simple_rnn_2/while/Identity_2:output:0"a
*sequential_2_simple_rnn_2_while_identity_33sequential_2/simple_rnn_2/while/Identity_3:output:0"a
*sequential_2_simple_rnn_2_while_identity_43sequential_2/simple_rnn_2/while/Identity_4:output:0"?
Isequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_strided_slice_1Ksequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_strided_slice_1_0"?
Qsequential_2_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceSsequential_2_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Rsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceTsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Psequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceRsequential_2_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
?sequential_2_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor?sequential_2_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2?
Hsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpHsequential_2/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Gsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpGsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Isequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpIsequential_2/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_21923
simple_rnn_2_input%
simple_rnn_2_21910:	?!
simple_rnn_2_21912:	?&
simple_rnn_2_21914:
?? 
dense_3_21917:	?
dense_3_21919:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_21910simple_rnn_2_21912simple_rnn_2_21914*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_218212&
$simple_rnn_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_3_21917dense_3_21919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_216662!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?c
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_22182

inputsP
=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:	?M
>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:	?S
?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:
??9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?simple_rnn_2/while^
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_2/Shape?
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_2/strided_slice/stack?
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_1?
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_2?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slicew
simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/mul/y?
simple_rnn_2/zeros/mulMul#simple_rnn_2/strided_slice:output:0!simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/muly
simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/Less/y?
simple_rnn_2/zeros/LessLesssimple_rnn_2/zeros/mul:z:0"simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/Less}
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/packed/1?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_2/zeros/packedy
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_2/zeros/Const?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_2/zeros?
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose/perm?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
simple_rnn_2/transposev
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_2/Shape_1?
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_1/stack?
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_1?
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_2?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slice_1?
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_2/TensorArrayV2/element_shape?
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2?
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_2/stack?
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_1?
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_2?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_2?
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?
%simple_rnn_2/simple_rnn_cell_2/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_2/simple_rnn_cell_2/MatMul?
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
&simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0=simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_2/simple_rnn_cell_2/BiasAdd?
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
'simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_2/simple_rnn_cell_2/MatMul_1?
"simple_rnn_2/simple_rnn_cell_2/addAddV2/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_2/simple_rnn_cell_2/add?
#simple_rnn_2/simple_rnn_cell_2/TanhTanh&simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_2/simple_rnn_cell_2/Tanh?
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_2/TensorArrayV2_1/element_shape?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2_1h
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_2/time?
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_2/while/maximum_iterations?
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_2/while/loop_counter?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_2_while_body_22110*)
cond!R
simple_rnn_2_while_cond_22109*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_2/while?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype021
/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_2/strided_slice_3/stack?
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_2/strided_slice_3/stack_1?
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_3/stack_2?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_3?
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose_1/perm?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_2/transpose_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul%simple_rnn_2/strided_slice_3:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp^simple_rnn_2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21298

inputs*
simple_rnn_cell_2_21223:	?&
simple_rnn_cell_2_21225:	?+
simple_rnn_cell_2_21227:
??
identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_21223simple_rnn_cell_2_21225simple_rnn_cell_2_21227*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_211682+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_21223simple_rnn_cell_2_21225simple_rnn_cell_2_21227*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21235*
condR
while_cond_21234*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_21060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21060___redundant_placeholder03
/while_while_cond_21060___redundant_placeholder13
/while_while_cond_21060___redundant_placeholder23
/while_while_cond_21060___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?C
?
simple_rnn_2_while_body_221106
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0X
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?U
Fsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?[
Gsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorV
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:	?S
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?Y
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
+simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_2/while/simple_rnn_cell_2/MatMul?
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
,simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_2/while/simple_rnn_cell_2/BiasAdd?
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1?
(simple_rnn_2/while/simple_rnn_cell_2/addAddV25simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_2/while/simple_rnn_cell_2/add?
)simple_rnn_2/while/simple_rnn_cell_2/TanhTanh,simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_2/while/simple_rnn_cell_2/Tanh?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add/y?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/addz
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add_1/y?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/add_1?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_1?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_2?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_3?
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_2/while/Identity_4"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*sequential_2_simple_rnn_2_while_cond_20923P
Lsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_loop_counterV
Rsequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_maximum_iterations/
+sequential_2_simple_rnn_2_while_placeholder1
-sequential_2_simple_rnn_2_while_placeholder_11
-sequential_2_simple_rnn_2_while_placeholder_2R
Nsequential_2_simple_rnn_2_while_less_sequential_2_simple_rnn_2_strided_slice_1g
csequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_cond_20923___redundant_placeholder0g
csequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_cond_20923___redundant_placeholder1g
csequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_cond_20923___redundant_placeholder2g
csequential_2_simple_rnn_2_while_sequential_2_simple_rnn_2_while_cond_20923___redundant_placeholder3,
(sequential_2_simple_rnn_2_while_identity
?
$sequential_2/simple_rnn_2/while/LessLess+sequential_2_simple_rnn_2_while_placeholderNsequential_2_simple_rnn_2_while_less_sequential_2_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_2/simple_rnn_2/while/Less?
(sequential_2/simple_rnn_2/while/IdentityIdentity(sequential_2/simple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_2/simple_rnn_2/while/Identity"]
(sequential_2_simple_rnn_2_while_identity1sequential_2/simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22436
inputs_0C
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22370*
condR
while_cond_22369*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22740

inputs
states_01
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22660

inputsC
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22594*
condR
while_cond_22593*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_21946
simple_rnn_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_209962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_21863

inputs%
simple_rnn_2_21850:	?!
simple_rnn_2_21852:	?&
simple_rnn_2_21854:
?? 
dense_3_21857:	?
dense_3_21859:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_21850simple_rnn_2_21852simple_rnn_2_21854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_218212&
$simple_rnn_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_3_21857dense_3_21859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_216662!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
while_body_21061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_2_21083_0:	?.
while_simple_rnn_cell_2_21085_0:	?3
while_simple_rnn_cell_2_21087_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_2_21083:	?,
while_simple_rnn_cell_2_21085:	?1
while_simple_rnn_cell_2_21087:
????/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_2_21083_0while_simple_rnn_cell_2_21085_0while_simple_rnn_cell_2_21087_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_2104821
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:10^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_2_21083while_simple_rnn_cell_2_21083_0"@
while_simple_rnn_cell_2_21085while_simple_rnn_cell_2_21085_0"@
while_simple_rnn_cell_2_21087while_simple_rnn_cell_2_21087_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_simple_rnn_2_layer_call_fn_22693

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_216482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_21907
simple_rnn_2_input%
simple_rnn_2_21894:	?!
simple_rnn_2_21896:	?&
simple_rnn_2_21898:
?? 
dense_3_21901:	?
dense_3_21903:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_21894simple_rnn_2_21896simple_rnn_2_21898*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_216482&
$simple_rnn_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_3_21901dense_3_21903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_216662!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22324
inputs_0C
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22258*
condR
while_cond_22257*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?c
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_22064

inputsP
=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:	?M
>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:	?S
?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:
??9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?simple_rnn_2/while^
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_2/Shape?
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_2/strided_slice/stack?
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_1?
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_2?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slicew
simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/mul/y?
simple_rnn_2/zeros/mulMul#simple_rnn_2/strided_slice:output:0!simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/muly
simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/Less/y?
simple_rnn_2/zeros/LessLesssimple_rnn_2/zeros/mul:z:0"simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/Less}
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/packed/1?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_2/zeros/packedy
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_2/zeros/Const?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_2/zeros?
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose/perm?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
simple_rnn_2/transposev
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_2/Shape_1?
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_1/stack?
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_1?
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_2?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slice_1?
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_2/TensorArrayV2/element_shape?
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2?
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_2/stack?
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_1?
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_2?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_2?
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?
%simple_rnn_2/simple_rnn_cell_2/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_2/simple_rnn_cell_2/MatMul?
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
&simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0=simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_2/simple_rnn_cell_2/BiasAdd?
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
'simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_2/simple_rnn_cell_2/MatMul_1?
"simple_rnn_2/simple_rnn_cell_2/addAddV2/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_2/simple_rnn_cell_2/add?
#simple_rnn_2/simple_rnn_cell_2/TanhTanh&simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_2/simple_rnn_cell_2/Tanh?
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_2/TensorArrayV2_1/element_shape?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2_1h
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_2/time?
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_2/while/maximum_iterations?
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_2/while/loop_counter?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_2_while_body_21992*)
cond!R
simple_rnn_2_while_cond_21991*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_2/while?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype021
/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_2/strided_slice_3/stack?
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_2/strided_slice_3/stack_1?
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_3/stack_2?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_3?
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose_1/perm?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_2/transpose_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul%simple_rnn_2/strided_slice_3:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp^simple_rnn_2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
simple_rnn_2_while_cond_219916
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_21991___redundant_placeholder0M
Isimple_rnn_2_while_simple_rnn_2_while_cond_21991___redundant_placeholder1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_21991___redundant_placeholder2M
Isimple_rnn_2_while_simple_rnn_2_while_cond_21991___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_2/while/Less?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_2/while/Identity"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
1__inference_simple_rnn_cell_2_layer_call_fn_22785

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_211682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?4
?
while_body_22258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_21048

inputs

states1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
'__inference_dense_3_layer_call_fn_22723

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_216662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_2_layer_call_fn_22682
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_212982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?M
?
!__inference__traced_restore_22920
file_prefix2
assignvariableop_dense_3_kernel:	?-
assignvariableop_1_dense_3_bias:)
assignvariableop_2_rmsprop_iter:	 *
 assignvariableop_3_rmsprop_decay: 2
(assignvariableop_4_rmsprop_learning_rate: -
#assignvariableop_5_rmsprop_momentum: (
assignvariableop_6_rmsprop_rho: K
8assignvariableop_7_simple_rnn_2_simple_rnn_cell_2_kernel:	?V
Bassignvariableop_8_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel:
??E
6assignvariableop_9_simple_rnn_2_simple_rnn_cell_2_bias:	?#
assignvariableop_10_total: #
assignvariableop_11_count: A
.assignvariableop_12_rmsprop_dense_3_kernel_rms:	?:
,assignvariableop_13_rmsprop_dense_3_bias_rms:X
Eassignvariableop_14_rmsprop_simple_rnn_2_simple_rnn_cell_2_kernel_rms:	?c
Oassignvariableop_15_rmsprop_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_rms:
??R
Cassignvariableop_16_rmsprop_simple_rnn_2_simple_rnn_cell_2_bias_rms:	?
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_2_simple_rnn_cell_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_2_simple_rnn_cell_2_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_2_simple_rnn_cell_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_rmsprop_dense_3_kernel_rmsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_rmsprop_dense_3_bias_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpEassignvariableop_14_rmsprop_simple_rnn_2_simple_rnn_cell_2_kernel_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpOassignvariableop_15_rmsprop_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpCassignvariableop_16_rmsprop_simple_rnn_2_simple_rnn_cell_2_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17?
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?4
?
while_body_21755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_22593
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22593___redundant_placeholder03
/while_while_cond_22593___redundant_placeholder13
/while_while_cond_22593___redundant_placeholder23
/while_while_cond_22593___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_21673

inputs%
simple_rnn_2_21649:	?!
simple_rnn_2_21651:	?&
simple_rnn_2_21653:
?? 
dense_3_21667:	?
dense_3_21669:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_21649simple_rnn_2_21651simple_rnn_2_21653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_216482&
$simple_rnn_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_3_21667dense_3_21669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_216662!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
simple_rnn_2_while_cond_221096
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_22109___redundant_placeholder0M
Isimple_rnn_2_while_simple_rnn_2_while_cond_22109___redundant_placeholder1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_22109___redundant_placeholder2M
Isimple_rnn_2_while_simple_rnn_2_while_cond_22109___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_2/while/Less?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_2/while/Identity"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_22369
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22369___redundant_placeholder03
/while_while_cond_22369___redundant_placeholder13
/while_while_cond_22369___redundant_placeholder23
/while_while_cond_22369___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?C
?
simple_rnn_2_while_body_219926
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0X
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?U
Fsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?[
Gsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorV
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:	?S
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?Y
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
+simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_2/while/simple_rnn_cell_2/MatMul?
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
,simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_2/while/simple_rnn_cell_2/BiasAdd?
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1?
(simple_rnn_2/while/simple_rnn_cell_2/addAddV25simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_2/while/simple_rnn_cell_2/add?
)simple_rnn_2/while/simple_rnn_cell_2/TanhTanh,simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_2/while/simple_rnn_cell_2/Tanh?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add/y?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/addz
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add_1/y?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/add_1?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_1?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_2?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_3?
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_2/while/Identity_4"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?.
?
__inference__traced_save_22859
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopD
@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopN
Jsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_3_bias_rms_read_readvariableopP
Lsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_kernel_rms_read_readvariableopZ
Vsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopJsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableop>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop3savev2_rmsprop_dense_3_bias_rms_read_readvariableopLsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_kernel_rms_read_readvariableopVsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_rms_read_readvariableopJsavev2_rmsprop_simple_rnn_2_simple_rnn_cell_2_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapesr
p: :	?:: : : : : :	?:
??:?: : :	?::	?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?4
?
while_body_22482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
1__inference_simple_rnn_cell_2_layer_call_fn_22771

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_210482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?$
?
while_body_21235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_2_21257_0:	?.
while_simple_rnn_cell_2_21259_0:	?3
while_simple_rnn_cell_2_21261_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_2_21257:	?,
while_simple_rnn_cell_2_21259:	?1
while_simple_rnn_cell_2_21261:
????/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_2_21257_0while_simple_rnn_cell_2_21259_0while_simple_rnn_cell_2_21261_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_2116821
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:10^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_2_21257while_simple_rnn_cell_2_21257_0"@
while_simple_rnn_cell_2_21259while_simple_rnn_cell_2_21259_0"@
while_simple_rnn_cell_2_21261while_simple_rnn_cell_2_21261_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?=
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21124

inputs*
simple_rnn_cell_2_21049:	?&
simple_rnn_cell_2_21051:	?+
simple_rnn_cell_2_21053:
??
identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_21049simple_rnn_cell_2_21051simple_rnn_cell_2_21053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_210482+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_21049simple_rnn_cell_2_21051simple_rnn_cell_2_21053*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21061*
condR
while_cond_21060*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_2/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_21754
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21754___redundant_placeholder03
/while_while_cond_21754___redundant_placeholder13
/while_while_cond_21754___redundant_placeholder23
/while_while_cond_21754___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?4
?
while_body_22594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:	?H
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:	?N
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_2_matmul_readvariableop_resource:	?F
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:	?L
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:
????.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_21891
simple_rnn_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_218632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
while_cond_21234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21234___redundant_placeholder03
/while_while_cond_21234___redundant_placeholder13
/while_while_cond_21234___redundant_placeholder23
/while_while_cond_21234___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_21581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21581___redundant_placeholder03
/while_while_cond_21581___redundant_placeholder13
/while_while_cond_21581___redundant_placeholder23
/while_while_cond_21581___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22548

inputsC
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22482*
condR
while_cond_22481*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22757

inputs
states_01
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21821

inputsC
0simple_rnn_cell_2_matmul_readvariableop_resource:	?@
1simple_rnn_cell_2_biasadd_readvariableop_resource:	?F
2simple_rnn_cell_2_matmul_1_readvariableop_resource:
??
identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_21755*
condR
while_cond_21754*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_22481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22481___redundant_placeholder03
/while_while_cond_22481___redundant_placeholder13
/while_while_cond_22481___redundant_placeholder23
/while_while_cond_22481___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?w
?
 __inference__wrapped_model_20996
simple_rnn_2_input]
Jsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:	?Z
Ksequential_2_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:	?`
Lsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:
??F
3sequential_2_dense_3_matmul_readvariableop_resource:	?B
4sequential_2_dense_3_biasadd_readvariableop_resource:
identity??+sequential_2/dense_3/BiasAdd/ReadVariableOp?*sequential_2/dense_3/MatMul/ReadVariableOp?Bsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Asequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?Csequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?sequential_2/simple_rnn_2/while?
sequential_2/simple_rnn_2/ShapeShapesimple_rnn_2_input*
T0*
_output_shapes
:2!
sequential_2/simple_rnn_2/Shape?
-sequential_2/simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_2/simple_rnn_2/strided_slice/stack?
/sequential_2/simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn_2/strided_slice/stack_1?
/sequential_2/simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn_2/strided_slice/stack_2?
'sequential_2/simple_rnn_2/strided_sliceStridedSlice(sequential_2/simple_rnn_2/Shape:output:06sequential_2/simple_rnn_2/strided_slice/stack:output:08sequential_2/simple_rnn_2/strided_slice/stack_1:output:08sequential_2/simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_2/simple_rnn_2/strided_slice?
%sequential_2/simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_2/simple_rnn_2/zeros/mul/y?
#sequential_2/simple_rnn_2/zeros/mulMul0sequential_2/simple_rnn_2/strided_slice:output:0.sequential_2/simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_2/simple_rnn_2/zeros/mul?
&sequential_2/simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_2/simple_rnn_2/zeros/Less/y?
$sequential_2/simple_rnn_2/zeros/LessLess'sequential_2/simple_rnn_2/zeros/mul:z:0/sequential_2/simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_2/simple_rnn_2/zeros/Less?
(sequential_2/simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_2/simple_rnn_2/zeros/packed/1?
&sequential_2/simple_rnn_2/zeros/packedPack0sequential_2/simple_rnn_2/strided_slice:output:01sequential_2/simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/simple_rnn_2/zeros/packed?
%sequential_2/simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_2/simple_rnn_2/zeros/Const?
sequential_2/simple_rnn_2/zerosFill/sequential_2/simple_rnn_2/zeros/packed:output:0.sequential_2/simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_2/simple_rnn_2/zeros?
(sequential_2/simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_2/simple_rnn_2/transpose/perm?
#sequential_2/simple_rnn_2/transpose	Transposesimple_rnn_2_input1sequential_2/simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2%
#sequential_2/simple_rnn_2/transpose?
!sequential_2/simple_rnn_2/Shape_1Shape'sequential_2/simple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_2/simple_rnn_2/Shape_1?
/sequential_2/simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_2/simple_rnn_2/strided_slice_1/stack?
1sequential_2/simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_2/simple_rnn_2/strided_slice_1/stack_1?
1sequential_2/simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_2/simple_rnn_2/strided_slice_1/stack_2?
)sequential_2/simple_rnn_2/strided_slice_1StridedSlice*sequential_2/simple_rnn_2/Shape_1:output:08sequential_2/simple_rnn_2/strided_slice_1/stack:output:0:sequential_2/simple_rnn_2/strided_slice_1/stack_1:output:0:sequential_2/simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_2/simple_rnn_2/strided_slice_1?
5sequential_2/simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_2/simple_rnn_2/TensorArrayV2/element_shape?
'sequential_2/simple_rnn_2/TensorArrayV2TensorListReserve>sequential_2/simple_rnn_2/TensorArrayV2/element_shape:output:02sequential_2/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_2/simple_rnn_2/TensorArrayV2?
Osequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Q
Osequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
Asequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_2/simple_rnn_2/transpose:y:0Xsequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
/sequential_2/simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_2/simple_rnn_2/strided_slice_2/stack?
1sequential_2/simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_2/simple_rnn_2/strided_slice_2/stack_1?
1sequential_2/simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_2/simple_rnn_2/strided_slice_2/stack_2?
)sequential_2/simple_rnn_2/strided_slice_2StridedSlice'sequential_2/simple_rnn_2/transpose:y:08sequential_2/simple_rnn_2/strided_slice_2/stack:output:0:sequential_2/simple_rnn_2/strided_slice_2/stack_1:output:0:sequential_2/simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)sequential_2/simple_rnn_2/strided_slice_2?
Asequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02C
Asequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp?
2sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMulMatMul2sequential_2/simple_rnn_2/strided_slice_2:output:0Isequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul?
Bsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKsequential_2_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3sequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd<sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0Jsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3sequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd?
Csequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02E
Csequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMul(sequential_2/simple_rnn_2/zeros:output:0Ksequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1?
/sequential_2/simple_rnn_2/simple_rnn_cell_2/addAddV2<sequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:0>sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:??????????21
/sequential_2/simple_rnn_2/simple_rnn_cell_2/add?
0sequential_2/simple_rnn_2/simple_rnn_cell_2/TanhTanh3sequential_2/simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*(
_output_shapes
:??????????22
0sequential_2/simple_rnn_2/simple_rnn_cell_2/Tanh?
7sequential_2/simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7sequential_2/simple_rnn_2/TensorArrayV2_1/element_shape?
)sequential_2/simple_rnn_2/TensorArrayV2_1TensorListReserve@sequential_2/simple_rnn_2/TensorArrayV2_1/element_shape:output:02sequential_2/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_2/simple_rnn_2/TensorArrayV2_1?
sequential_2/simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_2/simple_rnn_2/time?
2sequential_2/simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_2/simple_rnn_2/while/maximum_iterations?
,sequential_2/simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_2/simple_rnn_2/while/loop_counter?
sequential_2/simple_rnn_2/whileWhile5sequential_2/simple_rnn_2/while/loop_counter:output:0;sequential_2/simple_rnn_2/while/maximum_iterations:output:0'sequential_2/simple_rnn_2/time:output:02sequential_2/simple_rnn_2/TensorArrayV2_1:handle:0(sequential_2/simple_rnn_2/zeros:output:02sequential_2/simple_rnn_2/strided_slice_1:output:0Qsequential_2/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resourceKsequential_2_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resourceLsequential_2_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_2_simple_rnn_2_while_body_20924*6
cond.R,
*sequential_2_simple_rnn_2_while_cond_20923*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2!
sequential_2/simple_rnn_2/while?
Jsequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
<sequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_2/simple_rnn_2/while:output:3Ssequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02>
<sequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
/sequential_2/simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????21
/sequential_2/simple_rnn_2/strided_slice_3/stack?
1sequential_2/simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_2/simple_rnn_2/strided_slice_3/stack_1?
1sequential_2/simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_2/simple_rnn_2/strided_slice_3/stack_2?
)sequential_2/simple_rnn_2/strided_slice_3StridedSliceEsequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:08sequential_2/simple_rnn_2/strided_slice_3/stack:output:0:sequential_2/simple_rnn_2/strided_slice_3/stack_1:output:0:sequential_2/simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2+
)sequential_2/simple_rnn_2/strided_slice_3?
*sequential_2/simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_2/simple_rnn_2/transpose_1/perm?
%sequential_2/simple_rnn_2/transpose_1	TransposeEsequential_2/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:03sequential_2/simple_rnn_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2'
%sequential_2/simple_rnn_2/transpose_1?
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_2/dense_3/MatMul/ReadVariableOp?
sequential_2/dense_3/MatMulMatMul2sequential_2/simple_rnn_2/strided_slice_3:output:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_3/MatMul?
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_3/BiasAdd/ReadVariableOp?
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_3/BiasAdd?
IdentityIdentity%sequential_2/dense_3/BiasAdd:output:0,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOpC^sequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpD^sequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp ^sequential_2/simple_rnn_2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2?
Bsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpBsequential_2/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Asequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpAsequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Csequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpCsequential_2/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2B
sequential_2/simple_rnn_2/whilesequential_2/simple_rnn_2/while:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
,__inference_sequential_2_layer_call_fn_22212

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_218632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_21666

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_2_input?
$serving_default_simple_rnn_2_input:0?????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?$
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
@_default_save_signature
*A&call_and_return_all_conditional_losses
B__call__"?"
_tf_keras_sequential?"{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_2_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 10}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 20, 1]}, "float32", "simple_rnn_2_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_2_input"}, "shared_object_id": 0}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?
	cell


state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"?
_tf_keras_rnn_layer?{"name": "simple_rnn_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "SimpleRNN", "config": {"name": "simple_rnn_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 5, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 10}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
iter
	decay
learning_rate
momentum
rho	rms;	rms<	rms=	rms>	rms?"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
non_trainable_variables
metrics
	variables
layer_metrics
 layer_regularization_losses
regularization_losses
trainable_variables

!layers
B__call__
@_default_save_signature
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_layer?{"name": "simple_rnn_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SimpleRNNCell", "config": {"name": "simple_rnn_cell_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 4}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
&non_trainable_variables
'metrics
	variables
(layer_metrics

)states
*layer_regularization_losses
regularization_losses
trainable_variables

+layers
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
,non_trainable_variables
-metrics
	variables
.layer_metrics
/layer_regularization_losses
regularization_losses
trainable_variables

0layers
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
8:6	?2%simple_rnn_2/simple_rnn_cell_2/kernel
C:A
??2/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
2:0?2#simple_rnn_2/simple_rnn_cell_2/bias
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
2non_trainable_variables
3metrics
"	variables
4layer_metrics
5layer_regularization_losses
#regularization_losses
$trainable_variables

6layers
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	7total
	8count
9	variables
:	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 12}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
+:)	?2RMSprop/dense_3/kernel/rms
$:"2RMSprop/dense_3/bias/rms
B:@	?21RMSprop/simple_rnn_2/simple_rnn_cell_2/kernel/rms
M:K
??2;RMSprop/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/rms
<::?2/RMSprop/simple_rnn_2/simple_rnn_cell_2/bias/rms
?2?
 __inference__wrapped_model_20996?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
simple_rnn_2_input?????????
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_22064
G__inference_sequential_2_layer_call_and_return_conditional_losses_22182
G__inference_sequential_2_layer_call_and_return_conditional_losses_21907
G__inference_sequential_2_layer_call_and_return_conditional_losses_21923?
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
?2?
,__inference_sequential_2_layer_call_fn_21686
,__inference_sequential_2_layer_call_fn_22197
,__inference_sequential_2_layer_call_fn_22212
,__inference_sequential_2_layer_call_fn_21891?
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
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22324
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22436
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22548
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22660?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_simple_rnn_2_layer_call_fn_22671
,__inference_simple_rnn_2_layer_call_fn_22682
,__inference_simple_rnn_2_layer_call_fn_22693
,__inference_simple_rnn_2_layer_call_fn_22704?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_22714?
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
'__inference_dense_3_layer_call_fn_22723?
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
?B?
#__inference_signature_wrapper_21946simple_rnn_2_input"?
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
 
?2?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22740
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22757?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_simple_rnn_cell_2_layer_call_fn_22771
1__inference_simple_rnn_cell_2_layer_call_fn_22785?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_20996{??<
5?2
0?-
simple_rnn_2_input?????????
? "1?.
,
dense_3!?
dense_3??????????
B__inference_dense_3_layer_call_and_return_conditional_losses_22714]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_22723P0?-
&?#
!?
inputs??????????
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_21907wG?D
=?:
0?-
simple_rnn_2_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_21923wG?D
=?:
0?-
simple_rnn_2_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_22064k;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_22182k;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_21686jG?D
=?:
0?-
simple_rnn_2_input?????????
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_21891jG?D
=?:
0?-
simple_rnn_2_input?????????
p

 
? "???????????
,__inference_sequential_2_layer_call_fn_22197^;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_22212^;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_21946?U?R
? 
K?H
F
simple_rnn_2_input0?-
simple_rnn_2_input?????????"1?.
,
dense_3!?
dense_3??????????
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22324~O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#
?
0??????????
? ?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22436~O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#
?
0??????????
? ?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22548n??<
5?2
$?!
inputs?????????

 
p 

 
? "&?#
?
0??????????
? ?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_22660n??<
5?2
$?!
inputs?????????

 
p

 
? "&?#
?
0??????????
? ?
,__inference_simple_rnn_2_layer_call_fn_22671qO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "????????????
,__inference_simple_rnn_2_layer_call_fn_22682qO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "????????????
,__inference_simple_rnn_2_layer_call_fn_22693a??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
,__inference_simple_rnn_2_layer_call_fn_22704a??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22740?]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_22757?]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
1__inference_simple_rnn_cell_2_layer_call_fn_22771?]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
1__inference_simple_rnn_cell_2_layer_call_fn_22785?]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0??????????