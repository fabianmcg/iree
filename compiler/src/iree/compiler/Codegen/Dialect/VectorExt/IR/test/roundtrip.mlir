// RUN: iree-opt --split-input-file %s | FileCheck %s
// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

func.func @specify_inline_layout(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.to_layout %result to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [2, 4], outer_tile = [4, 1], thread_tile = [4, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 4]>) : vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-LABEL: func.func @specify_inline_layout
// CHECK:      iree_vector_ext.to_layout {{.*}} to layout({{.*}})

// -----

#nested_0 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 4],
  outer_tile = [4, 1],
  thread_tile = [4, 2],
  element_tile = [1, 4],

  subgroup_strides = [0, 0],
  thread_strides   = [1, 4]
>

#nested_1 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [4, 2],
  outer_tile = [1, 4],
  thread_tile = [2, 4],
  element_tile = [4, 1],

  subgroup_strides = [0, 0],
  thread_strides = [8, 2]
>

func.func @specify_nested(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {
    in_bounds = [true, true],
    layout0 = #nested_0,
    layout1 = #nested_1
  } : memref<32x32xf16>, vector<32x32xf16>
  return %result : vector<32x32xf16>
}

// CHECK: #[[$LAYOUT0:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroup_tile = [1, 1],
// CHECK-SAME: batch_tile = [2, 4],
// CHECK-SAME: outer_tile = [4, 1],
// CHECK-SAME: thread_tile = [4, 2],
// CHECK-SAME: element_tile = [1, 4],
// CHECK-SAME: subgroup_strides = [0, 0],
// CHECK-SAME: thread_strides = [1, 4]>

// CHECK: #[[$LAYOUT1:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroup_tile = [1, 1],
// CHECK-SAME: batch_tile = [4, 2],
// CHECK-SAME: outer_tile = [1, 4],
// CHECK-SAME: thread_tile = [2, 4],
// CHECK-SAME: element_tile = [4, 1],
// CHECK-SAME: subgroup_strides = [0, 0],
// CHECK-SAME: thread_strides = [8, 2]>

// CHECK-LABEL: func.func @specify_nested
// CHECK:      vector.transfer_read
// CHECK-SAME:         layout0 = #[[$LAYOUT0]]
// CHECK-SAME:         layout1 = #[[$LAYOUT1]]

// -----

#nested_0 = #iree_vector_ext.nested_layout<
  subgroup_tile = [],
  batch_tile = [],
  outer_tile = [],
  thread_tile = [],
  element_tile = [],

  subgroup_strides = [],
  thread_strides   = []
>

func.func @specify_nested_0d(%lhs: vector<f16>) -> vector<f16> {
  %result = iree_vector_ext.to_layout %lhs to layout(#nested_0) : vector<f16>
  func.return %result : vector<f16>
}

// CHECK: #[[$LAYOUT0:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroup_tile = [],
// CHECK-SAME: batch_tile = [],
// CHECK-SAME: outer_tile = [],
// CHECK-SAME: thread_tile = [],
// CHECK-SAME: element_tile = [],
// CHECK-SAME: subgroup_strides = [],
// CHECK-SAME: thread_strides = []>

// CHECK-LABEL: func.func @specify_nested_0d
// CHECK:      to_layout
// CHECK-SAME:         layout(#[[$LAYOUT0]])

// -----

func.func @to_simd_op(%simt: vector<4x4x4xf16>) -> vector<64x64xf16> {
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf16> -> vector<64x64xf16>
  func.return %simd : vector<64x64xf16>
}
// CHECK-LABEL: func.func @to_simd_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @to_simt_op(%simd: vector<64x64xf32>) -> vector<4x4x4xf32> {
  %simt = iree_vector_ext.to_simd %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  func.return %simt : vector<4x4x4xf32>
}
// CHECK-LABEL: func.func @to_simt_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @to_simd_op_0d(%simt: vector<f16>) -> vector<f16> {
  %simd = iree_vector_ext.to_simd %simt : vector<f16> -> vector<f16>
  func.return %simd : vector<f16>
}
// CHECK-LABEL: func.func @to_simd_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @to_simt_op_0d(%simd: vector<f32>) -> vector<f32> {
  %simt = iree_vector_ext.to_simd %simd : vector<f32> -> vector<f32>
  func.return %simt : vector<f32>
}
// CHECK-LABEL: func.func @to_simt_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @transfer_gather(%indices: vector<128xindex>,
  %indices1: vector<64xindex>,
  %indices2: vector<128x64xindex>,
  %source: tensor<4096x64xf16>)
  -> (vector<128x64xf16>, vector<128x64xf16>,vector<128x64xf16>,vector<128x64xf16>) {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // inner dimension gather
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, %indices1: vector<64xindex>], %cst0 { indexed_maps = [
                                             affine_map<(d0, d1) -> (d1)>]}
  : tensor<4096x64xf16>, vector<128x64xf16>

  // outer dimension gather
  %out1 = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices: vector<128xindex>, None], %cst0 { indexed_maps = [
                                                  affine_map<(d0, d1) -> (d0)>]}
  : tensor<4096x64xf16>, vector<128x64xf16>

  // full gather
  %out2 = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices: vector<128xindex>, %indices1: vector<64xindex>], %cst0
                                              { indexed_maps = [
                                                  affine_map<(d0, d1) -> (d0)>,
                                                  affine_map<(d0, d1) -> (d1)>]}
  : tensor<4096x64xf16>, vector<128x64xf16>

  // sparse gather
  %out3 = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, %indices2: vector<128x64xindex>], %cst0
                                              { indexed_maps = [
                                                  affine_map<(d0, d1) -> (d0, d1)>]}
  : tensor<4096x64xf16>, vector<128x64xf16>

  return %out, %out1, %out2, %out3 : vector<128x64xf16>, vector<128x64xf16>, vector<128x64xf16>, vector<128x64xf16>
}

// CHECK-LABEL: func.func @transfer_gather
// CHECK: iree_vector_ext.transfer_gather

// -----

// CHECK-LABEL: func.func @indexed_ops
func.func @indexed_ops(%buf0: memref<?x?xf32>, %buf1: memref<?x?x?xf32>,
    %b: index, %i: index, %j: index, %mask0: vector<4x4x4xi1>,
    %mask1: vector<4x4xi1>, %val: vector<4x4x4xf32>,
    %offsets0: vector<4x4x4xindex>, %offsets1: vector<4x4xindex>) {
  %v0 = iree_vector_ext.indexed_read %buf0[%i, %j]
      affine_map<()[b, ii, jj] -> (ii, jj)>
      mask %mask0 passthrough %val in_bounds [false, true] :
      memref<?x?xf32> -> vector<4x4x4xf32>

  %v1 = iree_vector_ext.indexed_read %buf0[%i, %j]
      affine_map<()[b, jj, ii] -> (ii, jj)>
      mask %mask0 passthrough %val in_bounds [true, false] :
      memref<?x?xf32> -> vector<4x4x4xf32>

  %v2 = iree_vector_ext.indexed_read %buf0[%i, %j] ins(%offsets0)
      affine_map<()[offset, b, ii, jj] -> (offset, jj)>
      mask %mask0 passthrough %val in_bounds [false, true] :
      memref<?x?xf32> -> vector<4x4x4xf32>

  %v3 = iree_vector_ext.indexed_read %buf0[%i, %j] ins(%offsets1)
      affine_map<()[offset, jj, ii] -> (offset, jj)>
      in_bounds [true, true] :
      memref<?x?xf32> -> vector<4x4xf32>
      
  iree_vector_ext.indexed_write %v3, %buf1[%b, %i, %j]
      affine_map<()[ii, jj] -> (0, ii, jj)>
      mask %mask1 in_bounds [false, true, true] :
      vector<4x4xf32>, memref<?x?x?xf32>

  iree_vector_ext.indexed_write %v3, %buf1[%b, %i, %j]
      affine_map<()[jj, ii] -> (0, ii, jj)>
      mask %mask1 in_bounds [true, false, true] :
      vector<4x4xf32>, memref<?x?x?xf32>

  iree_vector_ext.indexed_write %v3, %buf0[%i, %j] ins(%offsets1)
      affine_map<()[offset, jj, ii] -> (offset, jj)>
      in_bounds [true, true] :
      vector<4x4xf32>, memref<?x?xf32>
  return
}
