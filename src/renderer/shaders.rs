// File automatically generated by wgsl_bindgen^
//
// ^ wgsl_bindgen version 0.14.1
// Changes made to this file will not be saved.
// SourceHash: 2f65badd27d9a325d00931f7c560bafeb374efb2a626d1eef93c9139d9151545

#![allow(unused, non_snake_case, non_camel_case_types, non_upper_case_globals)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderEntry {
    Cube,
    Skybox,
}
impl ShaderEntry {
    pub fn create_pipeline_layout(&self, device: &wgpu::Device) -> wgpu::PipelineLayout {
        match self {
            Self::Cube => cube::create_pipeline_layout(device),
            Self::Skybox => skybox::create_pipeline_layout(device),
        }
    }
    pub fn create_shader_module_embed_source(
        &self,
        device: &wgpu::Device,
    ) -> wgpu::ShaderModule {
        match self {
            Self::Cube => cube::create_shader_module_embed_source(device),
            Self::Skybox => skybox::create_shader_module_embed_source(device),
        }
    }
}
mod _root {
    pub use super::*;
}
pub mod layout_asserts {
    use super::{_root, _root::*};
    const WGSL_BASE_TYPE_ASSERTS: () = {
        assert!(std::mem::size_of:: < glam::Vec3A > () == 16);
        assert!(std::mem::align_of:: < glam::Vec3A > () == 16);
        assert!(std::mem::size_of:: < glam::Vec4 > () == 16);
        assert!(std::mem::align_of:: < glam::Vec4 > () == 16);
        assert!(std::mem::size_of:: < glam::Mat3A > () == 48);
        assert!(std::mem::align_of:: < glam::Mat3A > () == 16);
        assert!(std::mem::size_of:: < glam::Mat4 > () == 64);
        assert!(std::mem::align_of:: < glam::Mat4 > () == 16);
    };
    const BGROUP_CAMERA_CAMERA_ASSERTS: () = {
        assert!(std::mem::offset_of!(bgroup_camera::Camera, view) == 0);
        assert!(std::mem::offset_of!(bgroup_camera::Camera, view_inv) == 64);
        assert!(std::mem::offset_of!(bgroup_camera::Camera, proj) == 128);
        assert!(std::mem::offset_of!(bgroup_camera::Camera, proj_inv) == 192);
        assert!(std::mem::size_of:: < bgroup_camera::Camera > () == 256);
    };
}
pub mod bgroup_camera {
    use super::{_root, _root::*};
    #[repr(C, align(16))]
    #[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
    pub struct Camera {
        /// size: 64, offset: 0x0, type: `mat4x4<f32>`
        pub view: glam::Mat4,
        /// size: 64, offset: 0x40, type: `mat4x4<f32>`
        pub view_inv: glam::Mat4,
        /// size: 64, offset: 0x80, type: `mat4x4<f32>`
        pub proj: glam::Mat4,
        /// size: 64, offset: 0xC0, type: `mat4x4<f32>`
        pub proj_inv: glam::Mat4,
    }
    impl Camera {
        pub const fn new(
            view: glam::Mat4,
            view_inv: glam::Mat4,
            proj: glam::Mat4,
            proj_inv: glam::Mat4,
        ) -> Self {
            Self {
                view,
                view_inv,
                proj,
                proj_inv,
            }
        }
    }
}
pub mod bytemuck_impls {
    use super::{_root, _root::*};
    unsafe impl bytemuck::Zeroable for bgroup_camera::Camera {}
    unsafe impl bytemuck::Pod for bgroup_camera::Camera {}
    unsafe impl bytemuck::Zeroable for cube::Vertex {}
    unsafe impl bytemuck::Pod for cube::Vertex {}
}
pub mod cube {
    use super::{_root, _root::*};
    #[repr(C)]
    #[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
    pub struct Vertex {
        pub position: glam::Vec4,
        pub tex_coord: [f32; 2],
    }
    impl Vertex {
        pub const fn new(position: glam::Vec4, tex_coord: [f32; 2]) -> Self {
            Self { position, tex_coord }
        }
    }
    impl Vertex {
        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: std::mem::offset_of!(Self, position) as u64,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: std::mem::offset_of!(Self, tex_coord) as u64,
                shader_location: 1,
            },
        ];
        pub const fn vertex_buffer_layout(
            step_mode: wgpu::VertexStepMode,
        ) -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Self>() as u64,
                step_mode,
                attributes: &Self::VERTEX_ATTRIBUTES,
            }
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup0EntriesParams<'a> {
        pub res_camera: wgpu::BufferBinding<'a>,
    }
    #[derive(Clone, Debug)]
    pub struct WgpuBindGroup0Entries<'a> {
        pub res_camera: wgpu::BindGroupEntry<'a>,
    }
    impl<'a> WgpuBindGroup0Entries<'a> {
        pub fn new(params: WgpuBindGroup0EntriesParams<'a>) -> Self {
            Self {
                res_camera: wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(params.res_camera),
                },
            }
        }
        pub fn as_array(self) -> [wgpu::BindGroupEntry<'a>; 1] {
            [self.res_camera]
        }
        pub fn collect<B: FromIterator<wgpu::BindGroupEntry<'a>>>(self) -> B {
            self.as_array().into_iter().collect()
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup0(wgpu::BindGroup);
    impl WgpuBindGroup0 {
        pub const LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube::BindGroup0::LayoutDescriptor"),
            entries: &[
                /// @binding(0): "_root::bgroup_camera::res_camera"
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<_root::bgroup_camera::Camera>() as _,
                        ),
                    },
                    count: None,
                },
            ],
        };
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&Self::LAYOUT_DESCRIPTOR)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: WgpuBindGroup0Entries,
        ) -> Self {
            let bind_group_layout = Self::get_bind_group_layout(&device);
            let entries = bindings.as_array();
            let bind_group = device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Cube::BindGroup0"),
                        layout: &bind_group_layout,
                        entries: &entries,
                    },
                );
            Self(bind_group)
        }
        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
            render_pass.set_bind_group(0, &self.0, &[]);
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup1EntriesParams<'a> {
        pub res_color: &'a wgpu::TextureView,
    }
    #[derive(Clone, Debug)]
    pub struct WgpuBindGroup1Entries<'a> {
        pub res_color: wgpu::BindGroupEntry<'a>,
    }
    impl<'a> WgpuBindGroup1Entries<'a> {
        pub fn new(params: WgpuBindGroup1EntriesParams<'a>) -> Self {
            Self {
                res_color: wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(params.res_color),
                },
            }
        }
        pub fn as_array(self) -> [wgpu::BindGroupEntry<'a>; 1] {
            [self.res_color]
        }
        pub fn collect<B: FromIterator<wgpu::BindGroupEntry<'a>>>(self) -> B {
            self.as_array().into_iter().collect()
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup1(wgpu::BindGroup);
    impl WgpuBindGroup1 {
        pub const LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube::BindGroup1::LayoutDescriptor"),
            entries: &[
                /// @binding(0): "res_color"
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        };
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&Self::LAYOUT_DESCRIPTOR)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: WgpuBindGroup1Entries,
        ) -> Self {
            let bind_group_layout = Self::get_bind_group_layout(&device);
            let entries = bindings.as_array();
            let bind_group = device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Cube::BindGroup1"),
                        layout: &bind_group_layout,
                        entries: &entries,
                    },
                );
            Self(bind_group)
        }
        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
            render_pass.set_bind_group(1, &self.0, &[]);
        }
    }
    #[derive(Debug, Copy, Clone)]
    pub struct WgpuBindGroups<'a> {
        pub bind_group0: &'a WgpuBindGroup0,
        pub bind_group1: &'a WgpuBindGroup1,
    }
    impl<'a> WgpuBindGroups<'a> {
        pub fn set(&self, pass: &mut wgpu::RenderPass<'a>) {
            self.bind_group0.set(pass);
            self.bind_group1.set(pass);
        }
    }
    pub fn set_bind_groups<'a>(
        pass: &mut wgpu::RenderPass<'a>,
        bind_group0: &'a WgpuBindGroup0,
        bind_group1: &'a WgpuBindGroup1,
    ) {
        bind_group0.set(pass);
        bind_group1.set(pass);
    }
    pub const ENTRY_VS_MESH: &str = "vs_mesh";
    pub const ENTRY_FS_MESH: &str = "fs_mesh";
    pub const ENTRY_FS_WIREFRAME: &str = "fs_wireframe";
    #[derive(Debug)]
    pub struct VertexEntry<const N: usize> {
        pub entry_point: &'static str,
        pub buffers: [wgpu::VertexBufferLayout<'static>; N],
        pub constants: std::collections::HashMap<String, f64>,
    }
    pub fn vertex_state<'a, const N: usize>(
        module: &'a wgpu::ShaderModule,
        entry: &'a VertexEntry<N>,
    ) -> wgpu::VertexState<'a> {
        wgpu::VertexState {
            module,
            entry_point: entry.entry_point,
            buffers: &entry.buffers,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &entry.constants,
                ..Default::default()
            },
        }
    }
    pub fn vs_mesh_entry(vertex: wgpu::VertexStepMode) -> VertexEntry<1> {
        VertexEntry {
            entry_point: ENTRY_VS_MESH,
            buffers: [Vertex::vertex_buffer_layout(vertex)],
            constants: Default::default(),
        }
    }
    #[derive(Debug)]
    pub struct FragmentEntry<const N: usize> {
        pub entry_point: &'static str,
        pub targets: [Option<wgpu::ColorTargetState>; N],
        pub constants: std::collections::HashMap<String, f64>,
    }
    pub fn fragment_state<'a, const N: usize>(
        module: &'a wgpu::ShaderModule,
        entry: &'a FragmentEntry<N>,
    ) -> wgpu::FragmentState<'a> {
        wgpu::FragmentState {
            module,
            entry_point: entry.entry_point,
            targets: &entry.targets,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &entry.constants,
                ..Default::default()
            },
        }
    }
    pub fn fs_mesh_entry(
        targets: [Option<wgpu::ColorTargetState>; 1],
    ) -> FragmentEntry<1> {
        FragmentEntry {
            entry_point: ENTRY_FS_MESH,
            targets,
            constants: Default::default(),
        }
    }
    pub fn fs_wireframe_entry(
        targets: [Option<wgpu::ColorTargetState>; 1],
    ) -> FragmentEntry<1> {
        FragmentEntry {
            entry_point: ENTRY_FS_WIREFRAME,
            targets,
            constants: Default::default(),
        }
    }
    #[derive(Debug)]
    pub struct WgpuPipelineLayout;
    impl WgpuPipelineLayout {
        pub fn bind_group_layout_entries(
            entries: [wgpu::BindGroupLayout; 2],
        ) -> [wgpu::BindGroupLayout; 2] {
            entries
        }
    }
    pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
        device
            .create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Cube::PipelineLayout"),
                    bind_group_layouts: &[
                        &WgpuBindGroup0::get_bind_group_layout(device),
                        &WgpuBindGroup1::get_bind_group_layout(device),
                    ],
                    push_constant_ranges: &[],
                },
            )
    }
    pub fn create_shader_module_embed_source(
        device: &wgpu::Device,
    ) -> wgpu::ShaderModule {
        let source = std::borrow::Cow::Borrowed(SHADER_STRING);
        device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube.wgsl"),
                source: wgpu::ShaderSource::Wgsl(source),
            })
    }
    pub const SHADER_STRING: &'static str = r#"
struct CameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
}

struct Vertex {
    @location(0) position: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
}

struct MeshInterp {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@group(0) @binding(0) 
var<uniform> res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX: CameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX;
@group(1) @binding(0) 
var res_color: texture_2d<u32>;

@vertex 
fn vs_mesh(in: Vertex) -> MeshInterp {
    var out: MeshInterp;

    out.tex_coord = in.tex_coord;
    let _e7 = res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX.proj;
    let _e10 = res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX.view;
    out.position = ((_e7 * _e10) * in.position);
    let _e14 = out;
    return _e14;
}

@fragment 
fn fs_mesh(vertex: MeshInterp) -> @location(0) vec4<f32> {
    let tex = textureLoad(res_color, vec2<i32>((vertex.tex_coord * 256f)), 0i);
    let v = (f32(tex.x) / 255f);
    return vec4<f32>((1f - (v * 5f)), (1f - (v * 15f)), (1f - (v * 50f)), 1f);
}

@fragment 
fn fs_wireframe(vertex_1: MeshInterp) -> @location(0) vec4<f32> {
    return vec4<f32>(0f, 0.5f, 0f, 0.5f);
}
"#;
}
pub mod skybox {
    use super::{_root, _root::*};
    #[derive(Debug)]
    pub struct WgpuBindGroup0EntriesParams<'a> {
        pub res_camera: wgpu::BufferBinding<'a>,
    }
    #[derive(Clone, Debug)]
    pub struct WgpuBindGroup0Entries<'a> {
        pub res_camera: wgpu::BindGroupEntry<'a>,
    }
    impl<'a> WgpuBindGroup0Entries<'a> {
        pub fn new(params: WgpuBindGroup0EntriesParams<'a>) -> Self {
            Self {
                res_camera: wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(params.res_camera),
                },
            }
        }
        pub fn as_array(self) -> [wgpu::BindGroupEntry<'a>; 1] {
            [self.res_camera]
        }
        pub fn collect<B: FromIterator<wgpu::BindGroupEntry<'a>>>(self) -> B {
            self.as_array().into_iter().collect()
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup0(wgpu::BindGroup);
    impl WgpuBindGroup0 {
        pub const LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox::BindGroup0::LayoutDescriptor"),
            entries: &[
                /// @binding(0): "_root::bgroup_camera::res_camera"
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<_root::bgroup_camera::Camera>() as _,
                        ),
                    },
                    count: None,
                },
            ],
        };
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&Self::LAYOUT_DESCRIPTOR)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: WgpuBindGroup0Entries,
        ) -> Self {
            let bind_group_layout = Self::get_bind_group_layout(&device);
            let entries = bindings.as_array();
            let bind_group = device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Skybox::BindGroup0"),
                        layout: &bind_group_layout,
                        entries: &entries,
                    },
                );
            Self(bind_group)
        }
        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
            render_pass.set_bind_group(0, &self.0, &[]);
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup1EntriesParams<'a> {
        pub res_texture: &'a wgpu::TextureView,
        pub res_sampler: &'a wgpu::Sampler,
    }
    #[derive(Clone, Debug)]
    pub struct WgpuBindGroup1Entries<'a> {
        pub res_texture: wgpu::BindGroupEntry<'a>,
        pub res_sampler: wgpu::BindGroupEntry<'a>,
    }
    impl<'a> WgpuBindGroup1Entries<'a> {
        pub fn new(params: WgpuBindGroup1EntriesParams<'a>) -> Self {
            Self {
                res_texture: wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(params.res_texture),
                },
                res_sampler: wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(params.res_sampler),
                },
            }
        }
        pub fn as_array(self) -> [wgpu::BindGroupEntry<'a>; 2] {
            [self.res_texture, self.res_sampler]
        }
        pub fn collect<B: FromIterator<wgpu::BindGroupEntry<'a>>>(self) -> B {
            self.as_array().into_iter().collect()
        }
    }
    #[derive(Debug)]
    pub struct WgpuBindGroup1(wgpu::BindGroup);
    impl WgpuBindGroup1 {
        pub const LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox::BindGroup1::LayoutDescriptor"),
            entries: &[
                /// @binding(0): "res_texture"
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                /// @binding(1): "res_sampler"
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        };
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&Self::LAYOUT_DESCRIPTOR)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: WgpuBindGroup1Entries,
        ) -> Self {
            let bind_group_layout = Self::get_bind_group_layout(&device);
            let entries = bindings.as_array();
            let bind_group = device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Skybox::BindGroup1"),
                        layout: &bind_group_layout,
                        entries: &entries,
                    },
                );
            Self(bind_group)
        }
        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
            render_pass.set_bind_group(1, &self.0, &[]);
        }
    }
    #[derive(Debug, Copy, Clone)]
    pub struct WgpuBindGroups<'a> {
        pub bind_group0: &'a WgpuBindGroup0,
        pub bind_group1: &'a WgpuBindGroup1,
    }
    impl<'a> WgpuBindGroups<'a> {
        pub fn set(&self, pass: &mut wgpu::RenderPass<'a>) {
            self.bind_group0.set(pass);
            self.bind_group1.set(pass);
        }
    }
    pub fn set_bind_groups<'a>(
        pass: &mut wgpu::RenderPass<'a>,
        bind_group0: &'a WgpuBindGroup0,
        bind_group1: &'a WgpuBindGroup1,
    ) {
        bind_group0.set(pass);
        bind_group1.set(pass);
    }
    pub const ENTRY_VS_SKYBOX: &str = "vs_skybox";
    pub const ENTRY_FS_SKYBOX: &str = "fs_skybox";
    #[derive(Debug)]
    pub struct VertexEntry<const N: usize> {
        pub entry_point: &'static str,
        pub buffers: [wgpu::VertexBufferLayout<'static>; N],
        pub constants: std::collections::HashMap<String, f64>,
    }
    pub fn vertex_state<'a, const N: usize>(
        module: &'a wgpu::ShaderModule,
        entry: &'a VertexEntry<N>,
    ) -> wgpu::VertexState<'a> {
        wgpu::VertexState {
            module,
            entry_point: entry.entry_point,
            buffers: &entry.buffers,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &entry.constants,
                ..Default::default()
            },
        }
    }
    pub fn vs_skybox_entry() -> VertexEntry<0> {
        VertexEntry {
            entry_point: ENTRY_VS_SKYBOX,
            buffers: [],
            constants: Default::default(),
        }
    }
    #[derive(Debug)]
    pub struct FragmentEntry<const N: usize> {
        pub entry_point: &'static str,
        pub targets: [Option<wgpu::ColorTargetState>; N],
        pub constants: std::collections::HashMap<String, f64>,
    }
    pub fn fragment_state<'a, const N: usize>(
        module: &'a wgpu::ShaderModule,
        entry: &'a FragmentEntry<N>,
    ) -> wgpu::FragmentState<'a> {
        wgpu::FragmentState {
            module,
            entry_point: entry.entry_point,
            targets: &entry.targets,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &entry.constants,
                ..Default::default()
            },
        }
    }
    pub fn fs_skybox_entry(
        targets: [Option<wgpu::ColorTargetState>; 1],
    ) -> FragmentEntry<1> {
        FragmentEntry {
            entry_point: ENTRY_FS_SKYBOX,
            targets,
            constants: Default::default(),
        }
    }
    #[derive(Debug)]
    pub struct WgpuPipelineLayout;
    impl WgpuPipelineLayout {
        pub fn bind_group_layout_entries(
            entries: [wgpu::BindGroupLayout; 2],
        ) -> [wgpu::BindGroupLayout; 2] {
            entries
        }
    }
    pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
        device
            .create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Skybox::PipelineLayout"),
                    bind_group_layouts: &[
                        &WgpuBindGroup0::get_bind_group_layout(device),
                        &WgpuBindGroup1::get_bind_group_layout(device),
                    ],
                    push_constant_ranges: &[],
                },
            )
    }
    pub fn create_shader_module_embed_source(
        device: &wgpu::Device,
    ) -> wgpu::ShaderModule {
        let source = std::borrow::Cow::Borrowed(SHADER_STRING);
        device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("skybox.wgsl"),
                source: wgpu::ShaderSource::Wgsl(source),
            })
    }
    pub const SHADER_STRING: &'static str = r#"
struct CameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
}

struct SkyboxInterp {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
}

@group(0) @binding(0) 
var<uniform> res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX: CameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX;
@group(1) @binding(0) 
var res_texture: texture_cube<f32>;
@group(1) @binding(1) 
var res_sampler: sampler;

@vertex 
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxInterp {
    var result: SkyboxInterp;

    let tmp1_ = (i32(vertex_index) / 2i);
    let tmp2_ = (i32(vertex_index) & 1i);
    let pos = vec4<f32>(((f32(tmp1_) * 4f) - 1f), ((f32(tmp2_) * 4f) - 1f), 1f, 1f);
    result.position = pos;
    let _e24 = res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX.proj_inv;
    let dir = vec4<f32>((_e24 * pos).xyz, 0f);
    let _e32 = res_cameraX_naga_oil_mod_XMJTXE33VOBPWGYLNMVZGCX.view_inv;
    result.tex_coord = (_e32 * dir).xyz;
    let _e35 = result;
    return _e35;
}

@fragment 
fn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4<f32> {
    let _e4 = textureSample(res_texture, res_sampler, vertex.tex_coord);
    return _e4;
}
"#;
}
