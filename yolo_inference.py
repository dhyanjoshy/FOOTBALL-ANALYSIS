from ultralytics import YOLO 

model = YOLO('models/best2.pt')
model.to('cuda')

results = model.predict('input_videos/08fd33_4.mp4',save=True, device = 'cuda')
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)