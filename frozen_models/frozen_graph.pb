
;
xPlaceholder*
shape:?????????*
dtype0
k
*model/dense/MatMul/ReadVariableOp/resourceConst*)
value B"?@?d?@)?|@??@*
dtype0
b
!model/dense/MatMul/ReadVariableOpIdentity*model/dense/MatMul/ReadVariableOp/resource*
T0
q
model/dense/MatMulMatMulx!model/dense/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( 
\
+model/dense/BiasAdd/ReadVariableOp/resourceConst*
valueB*H8?@*
dtype0
d
"model/dense/BiasAdd/ReadVariableOpIdentity+model/dense/BiasAdd/ReadVariableOp/resource*
T0
v
model/dense/BiasAddBiasAddmodel/dense/MatMul"model/dense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
2
IdentityIdentitymodel/dense/BiasAdd*
T0"?