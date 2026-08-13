import os
import shutil

def merge_datasets():
    old_train = 'dataset/Training'
    old_test = 'dataset/Testing'
    
    new_train = r'DATASET 2\extracted\train'
    new_test = r'DATASET 2\extracted\test'
    new_val = r'DATASET 2\extracted\val'
    
    out_train = 'dataset_combined/train'
    out_test = 'dataset_combined/test'
    
    # Map classes to a unified naming scheme
    class_mapping = {
        'glioma': 'glioma',
        'meningioma': 'meningioma',
        'notumor': 'no_tumor',
        'no_tumor': 'no_tumor',
        'pituitary': 'pituitary'
    }
    
    for split_dir in [out_train, out_test]:
        for class_name in set(class_mapping.values()):
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
    def copy_files(src_dir, dest_split):
        if not os.path.exists(src_dir):
            return
        for class_dir in os.listdir(src_dir):
            if class_dir not in class_mapping:
                continue
                
            unified_class = class_mapping[class_dir]
            src_class_path = os.path.join(src_dir, class_dir)
            dest_class_path = os.path.join(dest_split, unified_class)
            
            for file_name in os.listdir(src_class_path):
                src_file = os.path.join(src_class_path, file_name)
                # To prevent filename collisions, prepend a unique string based on the source dir
                prefix = os.path.basename(os.path.dirname(os.path.dirname(src_file))).replace(' ', '_')
                dest_file = os.path.join(dest_class_path, f"{prefix}_{file_name}")
                
                shutil.copy2(src_file, dest_file)
                
    print("Copying old training data...")
    copy_files(old_train, out_train)
    print("Copying old testing data...")
    copy_files(old_test, out_test)
    
    print("Copying new training data...")
    copy_files(new_train, out_train)
    print("Copying new validation data (to test split)...")
    copy_files(new_val, out_test)
    print("Copying new testing data...")
    copy_files(new_test, out_test)
    
    print("Merge complete!")

if __name__ == '__main__':
    merge_datasets()
