mod camera;

#[allow(clippy::all)]
mod shaders;

use std::sync::Arc;

use eframe::wgpu;
use glam::{Mat3, Mat4};
use gltf::mesh::Mode;
use puffin::profile_function;
use wgpu::{util::DeviceExt, BufferUsages};

use camera::ArcBallCamera;

use shaders::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

type CameraBindGroup = skybox::WgpuBindGroup0;
type CameraBindGroupEntries<'a> = skybox::WgpuBindGroup0Entries<'a>;
type CameraBindGroupEntriesParams<'a> = skybox::WgpuBindGroup0EntriesParams<'a>;

type NodeBindGroup = scene::WgpuBindGroup1;
type NodeBindGroupEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
type NodeBindGroupEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

type SkyboxBindGroup = skybox::WgpuBindGroup1;
type SkyboxBindGroupEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
type SkyboxBindGroupEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;

struct Node {
    mesh_index: usize,
    bgroup: NodeBindGroup,
}

#[derive(Debug)]
struct Mesh {
    primitives: Vec<Primitive>,
}

#[derive(Debug)]
struct Primitive {
    pipeline: wgpu::RenderPipeline,
    attrib_buffers: Vec<AttribBuffer>,
    draw_count: u32,
    index_data: Option<PrimitiveIndexData>,
}

#[derive(Debug)]
struct AttribBuffer {
    buffer: Arc<wgpu::Buffer>,
    offset: wgpu::BufferAddress,
}

#[derive(Debug)]
struct PrimitiveIndexData {
    buffer: Arc<wgpu::Buffer>,
    format: wgpu::IndexFormat,
    offset: u64,
}

pub struct SceneRenderer {
    camera_buf: wgpu::Buffer,
    user_camera: ArcBallCamera,
    camera_bgroup: CameraBindGroup,

    skybox_bgroup: SkyboxBindGroup,
    skybox_pipeline: wgpu::RenderPipeline,

    nodes: Vec<Node>,
    meshes: Vec<Mesh>,
}

fn get_vertex_format(accessor: &gltf::Accessor) -> wgpu::VertexFormat {
    use gltf::accessor::{DataType, Dimensions};
    match (
        accessor.normalized(),
        accessor.data_type(),
        accessor.dimensions(),
    ) {
        (true, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Snorm8x2,
        (true, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Snorm8x4,
        (true, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Unorm8x2,
        (true, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Unorm8x4,
        (true, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Snorm16x2,
        (true, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Snorm16x4,
        (true, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Unorm16x2,
        (true, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Unorm16x4,
        (false, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Sint8x2,
        (false, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Sint8x4,
        (false, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Uint8x2,
        (false, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Uint8x4,
        (false, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Sint16x2,
        (false, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Sint16x4,
        (false, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Uint16x2,
        (false, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Uint16x4,
        (false, DataType::U32, Dimensions::Scalar) => wgpu::VertexFormat::Uint32,
        (false, DataType::U32, Dimensions::Vec2) => wgpu::VertexFormat::Uint32x2,
        (false, DataType::U32, Dimensions::Vec3) => wgpu::VertexFormat::Uint32x3,
        (false, DataType::U32, Dimensions::Vec4) => wgpu::VertexFormat::Uint32x4,
        (_, DataType::F32, Dimensions::Scalar) => wgpu::VertexFormat::Float32,
        (_, DataType::F32, Dimensions::Vec2) => wgpu::VertexFormat::Float32x2,
        (_, DataType::F32, Dimensions::Vec3) => wgpu::VertexFormat::Float32x3,
        (_, DataType::F32, Dimensions::Vec4) => wgpu::VertexFormat::Float32x4,
        _ => unimplemented!(),
    }
}

impl SceneRenderer {
    pub fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // Camera

        let user_camera = ArcBallCamera::default();

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&bgroup_camera::Camera {
                view: Default::default(),
                view_inv: Default::default(),
                proj: Default::default(),
                proj_inv: Default::default(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bgroup = CameraBindGroup::from_bindings(
            device,
            CameraBindGroupEntries::new(CameraBindGroupEntriesParams {
                res_camera: camera_buf.as_entire_buffer_binding(),
            }),
        );

        // Skybox

        let ktx_reader = ktx2::Reader::new(include_bytes!("../assets/rgba8.ktx2"))
            .expect("Failed to find skybox texture");
        let mut image = Vec::with_capacity(ktx_reader.data().len());
        for level in ktx_reader.levels() {
            image.extend_from_slice(level);
        }
        let ktx_header = ktx_reader.header();
        let skybox_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("../assets/rgba8.ktx2"),
                size: wgpu::Extent3d {
                    width: ktx_header.pixel_width,
                    height: ktx_header.pixel_height,
                    depth_or_array_layers: ktx_header.face_count,
                },
                mip_level_count: ktx_header.level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &image,
        );
        let skybox_tview = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("../assets/rgba8.ktx2"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });

        let skybox_bgroup = SkyboxBindGroup::from_bindings(
            device,
            SkyboxBindGroupEntries::new(SkyboxBindGroupEntriesParams {
                res_texture: &skybox_tview,
                res_sampler: &device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("skybox sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
            }),
        );

        let shader = skybox::create_shader_module_embed_source(device);
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox"),
            layout: Some(&skybox::create_pipeline_layout(device)),
            vertex: skybox::vertex_state(&shader, &skybox::vs_skybox_entry()),
            fragment: Some(skybox::fragment_state(
                &shader,
                &skybox::fs_skybox_entry([Some(color_format.into())]),
            )),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load the GLTF scene

        // TODO: I'm currently embedding the glb file directly into my binary because I'm too lazy to handle file
        // loading separately on native and wasm targets. Maybe I should use WASI or some other API like that.
        // Also, I should load these asynchronously
        let glb_file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            // "/assets/models/AntiqueCamera/glTF-Binary/AntiqueCamera.glb"
            "/assets/models/Buggy/glTF-Binary/Buggy.glb"
        ));
        let (doc, buffer_data, _image_data) =
            gltf::import_slice(glb_file).expect("Failed to load GLTF file");

        let buffers: Vec<_> = doc
            .views()
            .map(|view: gltf::buffer::View| {
                let data = &buffer_data[view.buffer().index()].0;
                let contents = &data[view.offset()..view.offset() + view.length()];
                let usage = BufferUsages::COPY_DST | BufferUsages::VERTEX | BufferUsages::INDEX;
                Arc::new(
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: view.name(),
                        contents,
                        usage,
                    }),
                )
            })
            .collect();

        // Build Nodes

        let nodes = generate_nodes(&doc, device);

        // Build Meshes

        let meshes = generate_meshes(device, doc, buffers, color_format);

        Self {
            user_camera,
            camera_buf,
            camera_bgroup,

            skybox_bgroup,
            skybox_pipeline,

            nodes,
            meshes,
        }
    }

    pub fn prepare(
        &self,
        _device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
    ) -> Option<wgpu::CommandBuffer> {
        profile_function!();

        let view = self.user_camera.view_matrix();
        let proj = self.user_camera.projection_matrix();
        let camera = bgroup_camera::Camera {
            view,
            view_inv: view.inverse(),
            proj,
            proj_inv: proj.inverse(),
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&camera));

        None
    }

    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        profile_function!();

        self.camera_bgroup.set(rpass);

        for node in &self.nodes {
            node.bgroup.set(rpass);
            let mesh = self
                .meshes
                .get(node.mesh_index)
                .expect("Node didn't have a mesh");
            for primitive in &mesh.primitives {
                rpass.set_pipeline(&primitive.pipeline);
                for (i, attrib) in primitive.attrib_buffers.iter().enumerate() {
                    rpass.set_vertex_buffer(i as _, attrib.buffer.slice(attrib.offset..));
                }
                if let Some(index_data) = &primitive.index_data {
                    rpass.set_index_buffer(
                        index_data.buffer.slice(index_data.offset..),
                        index_data.format,
                    );
                    rpass.draw_indexed(0..primitive.draw_count, 0, 0..1);
                } else {
                    rpass.draw(0..primitive.draw_count, 0..1);
                }
            }
        }

        self.skybox_bgroup.set(rpass);
        rpass.set_pipeline(&self.skybox_pipeline);
        rpass.draw(0..3, 0..1);
    }

    pub fn run_ui(&mut self, ctx: &egui::Context) {
        profile_function!();

        if !ctx.wants_keyboard_input() && !ctx.wants_pointer_input() {
            ctx.input(|input| {
                self.user_camera.update(input);
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    // eframe doesn't support puffin on browser because it might not have a high resolution clock.
                    let mut are_scopes_on = puffin::are_scopes_on();
                    ui.toggle_value(&mut are_scopes_on, "Profiler");
                    puffin::set_scopes_on(are_scopes_on);
                }
                ui.menu_button("Camera", |ui| self.user_camera.run_ui(ui));
            });
        });
    }
}

fn generate_nodes(doc: &gltf::Document, device: &wgpu::Device) -> Vec<Node> {
    // Get world transforms
    let mut nodes_to_visit = Vec::new();
    for scene in doc.scenes() {
        nodes_to_visit.extend(scene.nodes().map(|n| (n, Mat4::IDENTITY)));
    }
    let mut world_transforms = vec![Mat4::IDENTITY; doc.nodes().len()];
    while let Some((node, parent_transform)) = nodes_to_visit.pop() {
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_transform = parent_transform * transform;
        world_transforms[node.index()] = world_transform;
        nodes_to_visit.extend(node.children().map(|n| (n, world_transform)));
    }
    let nodes = doc
        .nodes()
        .zip(world_transforms.iter())
        .filter_map(|(node, &transform)| {
            node.mesh().map(|mesh| {
                let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: node.name(),
                    contents: bytemuck::bytes_of(&scene::Node {
                        transform,
                        normal_transform: Mat4::from_mat3(
                            Mat3::from_mat4(transform).inverse().transpose(),
                        ),
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bgroup = NodeBindGroup::from_bindings(
                    device,
                    NodeBindGroupEntries::new(NodeBindGroupEntriesParams {
                        res_node: node_buf.as_entire_buffer_binding(),
                    }),
                );
                Node {
                    bgroup,
                    mesh_index: mesh.index(),
                }
            })
        })
        .collect();
    nodes
}

fn generate_meshes(
    device: &wgpu::Device,
    doc: gltf::Document,
    buffers: Vec<Arc<wgpu::Buffer>>,
    color_format: wgpu::TextureFormat,
) -> Vec<Mesh> {
    let shader = scene::create_shader_module_embed_source(device);

    doc.meshes()
        .map(|doc_mesh| {
            let primitives = doc_mesh
                .primitives()
                .map(|doc_primitive| {
                    let vertex_count = doc_primitive
                        .attributes()
                        .next()
                        .expect("There should be at least one attribute for each primitive")
                        .1
                        .count();
                    let (attrib_layouts, attrib_buffers): (Vec<_>, Vec<_>) = doc_primitive
                        .attributes()
                        .filter_map(|(semantic, accessor)| {
                            let shader_location = match semantic {
                                gltf::Semantic::Positions => 0,
                                gltf::Semantic::Normals => 1,
                                _ => return None,
                            };

                            let buffer_view =
                                accessor.view().expect("Accessor should have a buffer view");
                            let format = get_vertex_format(&accessor);
                            let stride = buffer_view
                                .stride()
                                .map(|s| s as u64)
                                .unwrap_or(format.size());
                            let (buf_offset, attrib_offset) = if accessor.offset() >= stride as _
                                || stride > device.limits().max_vertex_buffer_array_stride as _
                            {
                                (accessor.offset(), 0)
                            } else {
                                (0, accessor.offset())
                            };
                            Some((
                                (
                                    stride,
                                    wgpu::VertexAttribute {
                                        format,
                                        offset: attrib_offset as _,
                                        shader_location,
                                    },
                                ),
                                AttribBuffer {
                                    buffer: buffers[buffer_view.index()].clone(),
                                    offset: buf_offset as _,
                                },
                            ))
                        })
                        .unzip();

                    let attrib_buffer_layouts: Vec<_> = attrib_layouts
                        .iter()
                        .map(|(array_stride, attributes)| wgpu::VertexBufferLayout {
                            array_stride: *array_stride,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: std::slice::from_ref(attributes),
                        })
                        .collect();

                    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: doc_mesh.name(),
                        layout: Some(&scene::create_pipeline_layout(device)),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some(scene::ENTRY_VS_SCENE),
                            compilation_options: Default::default(),
                            buffers: &attrib_buffer_layouts,
                        },
                        fragment: Some(scene::fragment_state(
                            &shader,
                            &scene::fs_scene_entry([Some(color_format.into())]),
                        )),
                        primitive: wgpu::PrimitiveState {
                            topology: match doc_primitive.mode() {
                                Mode::Points => wgpu::PrimitiveTopology::PointList,
                                Mode::Lines => wgpu::PrimitiveTopology::LineList,
                                Mode::LineStrip => wgpu::PrimitiveTopology::LineStrip,
                                Mode::Triangles => wgpu::PrimitiveTopology::TriangleList,
                                Mode::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                                mode => unimplemented!("format {:?} not supported", mode),
                            },
                            cull_mode: Some(wgpu::Face::Back),
                            front_face: wgpu::FrontFace::Ccw,
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: DEPTH_FORMAT,
                            depth_write_enabled: true,
                            depth_compare: wgpu::CompareFunction::Less,
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                        cache: None,
                    });

                    let mut draw_count = vertex_count as u32;
                    let index_data = doc_primitive.indices().map(|indices| {
                        use gltf::accessor::DataType;
                        draw_count = indices.count() as _;
                        PrimitiveIndexData {
                            buffer: buffers[indices.view().unwrap().index()].clone(),
                            format: match indices.data_type() {
                                DataType::U16 => wgpu::IndexFormat::Uint16,
                                DataType::U32 => wgpu::IndexFormat::Uint32,
                                t => unimplemented!("Index type {:?} is not supported", t),
                            },
                            offset: indices.offset() as _,
                        }
                    });

                    Primitive {
                        pipeline,
                        attrib_buffers,
                        draw_count,
                        index_data,
                    }
                })
                .collect();
            Mesh { primitives }
        })
        .collect()
}
