import csv, os
os.makedirs('results/csv', exist_ok=True)

# 1. 阶段对比总表
stages = [
    # stage, platform, mAP50, mAP50_95, P, R, crazing_R, inference_ms, size_MB
    ['FP32','Desktop_4070Ti',0.736,0.409,0.645,0.667,0.278,'',''],
    ['ONNX','Desktop_4070Ti',0.710,0.375,0.653,0.645,0.218,3.5,12.0],
    ['FP16','Jetson_OrinNano',0.710,0.374,0.657,0.640,0.215,10.8,8.5],
    ['INT8','Jetson_OrinNano',0.635,0.312,0.536,0.659,0.316,9.4,4.7],
]
with open('results/csv/stage_comparison.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['stage','platform','mAP50','mAP50_95','precision','recall','crazing_recall','inference_ms','engine_size_MB'])
    w.writerows(stages)

# 2. 逐类别详细（每个阶段 x 每类 mAP50/P/R）
# 格式: stage, class, mAP50, precision, recall
perclass = [
    # FP32 (desktop baseline)
    ['FP32','crazing',0.411,0.434,0.278],
    ['FP32','inclusion',0.818,0.727,0.776],
    ['FP32','patches',0.928,0.795,0.860],
    ['FP32','pitted_surface',0.788,0.744,0.717],
    ['FP32','rolled-in_scale',0.632,0.636,0.522],
    ['FP32','scratches',0.838,0.535,0.846],
    # FP16 (jetson)
    ['FP16','crazing',0.346,0.420,0.215],
    ['FP16','inclusion',0.771,0.652,0.697],
    ['FP16','patches',0.923,0.792,0.887],
    ['FP16','pitted_surface',0.784,0.822,0.704],
    ['FP16','rolled-in_scale',0.610,0.694,0.494],
    ['FP16','scratches',0.825,0.564,0.844],
    # INT8 (jetson)
    ['INT8','crazing',0.345,0.438,0.316],
    ['INT8','inclusion',0.641,0.506,0.764],
    ['INT8','patches',0.917,0.751,0.879],
    ['INT8','pitted_surface',0.566,0.601,0.543],
    ['INT8','rolled-in_scale',0.574,0.533,0.595],
    ['INT8','scratches',0.763,0.384,0.859],
]
with open('results/csv/per_class.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['stage','class','mAP50','precision','recall'])
    w.writerows(perclass)

print('Wrote results/csv/stage_comparison.csv')
print('Wrote results/csv/per_class.csv')
