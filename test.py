import cv2
import numpy as np
import tensorflow as tf
import os
import argparse


MODEL_PATH = r"F:\eiott\Dyslexia_detection\synthdata\dyslexia_character_model.tflite"
CLASS_NAMES = ['Corrected', 'Normal', 'Reversal'] # Match your training output
INPUT_SIZE = (64, 64)

COLORS = {
    "Normal": (0, 255, 0),    # Green
    "Reversal": (0, 0, 255),   # Red
    "Corrected": (0, 255, 255) # Yellow
}

def preprocess_paragraph_image(image_path):
    """
    Loads an image and prepares three versions:
    1. inverted_gray: For the model to see (anti-aliased)
    2. binarized_img: For contour finding (hard edges)
    3. original_image: For drawing the results
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None
    
    original_image = img.copy()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    _, binarized_img = cv2.threshold(gray, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    
    inverted_gray = 255 - gray
    
    print(f"Image pre-processed.")
    return inverted_gray, binarized_img, original_image

def find_characters(binarized_img):
  
    contours, _ = cv2.findContours(binarized_img, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    

    MIN_AREA = 20
    MAX_AREA = 5000 
    
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
         
            x, y, w, h = cv2.boundingRect(cnt)
            
      
            padding = 4
            x_p = max(0, x - padding)
            y_p = max(0, y - padding)
            w_p = w + (padding * 2)
            h_p = h + (padding * 2)
            
            bounding_boxes.append((x_p, y_p, w_p, h_p))
            
    print(f"Found {len(bounding_boxes)} potential characters.")
    return bounding_boxes

def run_inference(image_path):
   
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
 
    inverted_gray, binarized_img, original_image = preprocess_paragraph_image(image_path)
    if original_image is None:
        return


    char_boxes = find_characters(binarized_img)
    
    # --- 4. Loop, Crop, and Classify ---
    counts = {name: 0 for name in CLASS_NAMES}
    
    for (x, y, w, h) in char_boxes:
  
        char_crop = inverted_gray[y:y+h, x:x+w]
        
        resized_char = cv2.resize(char_crop, INPUT_SIZE)
        
        input_data = np.expand_dims(resized_char, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)

     
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        else:
            input_data = input_data.astype(np.uint8)

    
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(prediction)
        
        counts[predicted_class] += 1
        
 
        x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(cv2.findContours(
            binarized_img[y:y+h, x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
        
        color = COLORS.get(predicted_class, (255, 255, 255))
       
        cv2.rectangle(original_image, (x + x_orig, y + y_orig), (x + x_orig + w_orig, y + y_orig + h_orig), color, 2)
        cv2.putText(original_image, f"{predicted_class[0]}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print("\n--- Test Complete ---")
    print("\nFinal Character Counts:")
    total = sum(counts.values())
    if total > 0:
        for name, count in counts.items():
            percentage = (count / total) * 100
            print(f"  - {name}: {count} ({percentage:.1f}%)")
    else:
        print("No characters were detected.")


    save_path = "test_result.png"
    cv2.imwrite(save_path, original_image)
    print(f"\nResult image saved to: {save_path}")
    
    try:
        cv2.imshow("Test Result", original_image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nCould not display image (running in a non-GUI environment?): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a handwriting image for dyslexia features.")
    parser.add_argument("-i", "--image", required=True, 
                        help="Path to the input image of handwritten text.")
    args = parser.parse_args()
    
    run_inference(args.image)