#resource "symbols/xauusd/models/16_04_2026-09_42__55-xau-default-fail/model.onnx" as uchar model_buffer[]

void OnStart() {
   Print(ArraySize(model_buffer));
}
