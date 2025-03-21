#![warn(clippy::all)]

mod app;
pub mod renderer;

use std::future::Future;

pub use app::RendererApp;

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn(task: impl Future<Output = ()> + 'static + Send) {
    tokio::spawn(task);
}

#[cfg(target_arch = "wasm32")]
pub fn spawn(task: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(task);
}
