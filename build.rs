fn main() {
    // Tell slint-build where your primary UI file is located.
    // Assuming you put it in a "ui" folder next to "src"
    slint_build::compile("ui/app.slint").unwrap();
}