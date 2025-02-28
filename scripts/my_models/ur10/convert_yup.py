import os

def change_zup_to_yup(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dae") or file.endswith(".DAE"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()

                # Replace 'Z_UP' with 'Y_UP'
                new_content = content.replace('<up_axis>Y_UP</up_axis>', '<up_axis>Z_UP</up_axis>')

                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write(new_content)

                print(f"Updated {file_path}")

directory = './sr_description'
change_zup_to_yup(directory)
