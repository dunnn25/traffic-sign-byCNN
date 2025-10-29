from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Tải model MỘT LẦN duy nhất khi ứng dụng khởi động
# Điều này giúp cải thiện đáng kể tốc độ dự đoán
try:
    MODEL = load_model('traffic_sign_model.h5')
    print("Traffic Sign Recognition model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None

# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }

def image_processing(img):
    if MODEL is None:
        return np.array([-1]) # Trả về lỗi nếu model chưa được tải

    data=[]
    image = Image.open(img)
    # Đảm bảo hình ảnh được chuyển thành mảng numpy và được chuẩn hóa nếu cần (thông thường là 0-255)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    
    # Bước 1: Chuẩn hóa dữ liệu đầu vào nếu mô hình được huấn luyện với dữ liệu chuẩn hóa
    # Nếu mô hình của bạn được huấn luyện trên dữ liệu chuẩn hóa (ví dụ: chia cho 255.0), 
    # bạn cần thêm dòng này: X_test = X_test / 255.0

    # Bước 2: Sử dụng model.predict() thay vì model.predict_classes() (đã bị loại bỏ)
    probabilities = MODEL.predict(X_test)
    
    # Bước 3: Tìm chỉ mục của lớp có xác suất cao nhất
    Y_pred = np.argmax(probabilities, axis=1) # Tìm index có giá trị lớn nhất theo chiều ngang (các lớp)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if MODEL is None:
        return "Error: Traffic Sign Recognition model is not loaded. Please check the path './model/TSR.h5'.", 500

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Tạo đường dẫn tạm thời an toàn
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        
        # Đảm bảo thư mục 'uploads' tồn tại
        os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)
        f.save(file_path)

        try:
            # Make prediction
            result = image_processing(file_path)
            s = [str(i) for i in result]
            a = int("".join(s))
            
            if a == -1: # Lỗi từ hàm image_processing
                result_text = "Prediction Error: Model failed to process the image."
            elif a in classes:
                result_text = "Predicted Traffic🚦Sign is: " + classes[a]
            else:
                result_text = "Predicted class index is invalid: " + str(a)

            return result_text
            
        except Exception as e:
            return f"An error occurred during prediction: {e}", 500
        finally:
            # Clean up: Xóa file đã tải lên sau khi xử lý
            if os.path.exists(file_path):
                os.remove(file_path)
            
    return None

if __name__ == '__main__':
    # Lưu ý: Trong môi trường production, KHÔNG nên chạy debug=True
    app.run(debug=True)