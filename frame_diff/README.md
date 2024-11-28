## Background
Computer vision: Object detection in video via frame differencing

## Environment
python 3.9.2

Use conda virtual envionment, no specific requirements for libraries' versions.

## Run
Locate in the script at first.

        python3 frame_diff_back.py --output_file test.csv --thresh 400 --video_path highway_cut.mp4

## Input
Three inputs, 2 optional: thresh, output_file ; 1 require: video_path
1. '--thresh', type=int, default=400, help='box size lower limit'
2. '--video_path', type=str, default='highway_cut.mp4', required=True, help='video_path end with .mp4'

   convert the video into frames into a temp folder 'Frame'

   e.g., ['Frame/frame_0.jpg', 'Frame/frame_1.jpg', 'Frame/frame_2.jpg']
   
   sample video is here: https://drive.google.com/file/d/17irhrdGFMfAPYE-iOPISItqRFgGHoAWB/view?usp=sharing
3. '--output_file', type=str, default='data.csv', help='box output file ending with csv'

## Output - a csv file with boxes coordinate
Outline: CSV/Images - Block/Image - Line/Box

Example: (example.csv; link: https://drive.google.com/file/d/11MzHxplylCONgWGr0lYxvV5zt-7EFXBZ/view?usp=drive_link)
1. One block refers to one image, separated by ';'.
2. Inside each block/image, one line refers to one box.
3. Inside each line/box, the format: [xmin, ymin, xmax, ymax].
4. The first 4 blocks/images are meaningless, as the model needs to take 4 previous images to do the background subtraction.
5. Empty block (no difference detected) is like: (no_diff.csv; link: https://drive.google.com/file/d/1WIGxVviA4mUAjF-9lgabvkc2AAbxqXE4/view?usp=drive_link)
6. Example code to read-in the csv file:

        # how to import the csv
        import numpy as np
    
        # Initialize an empty list to store arrays
        box_result = []
    
        # Read the CSV file
        with open('data.csv', 'r') as file:
            data = file.read().split(';\n')  # Split the file content by the word 'WORD'
    
            # Process each part of the split content
            for part in data:
                if part.strip():  # Check if the part is not empty
                    arr = np.genfromtxt(part.splitlines(), delimiter=',', dtype=int)
                    #box_result.append(arr)
                else:
                    arr = np.zeros((0, 4))#'No Diff'
                box_result.append(arr)
        # Print the list of arrays
        for arr in box_result:
            print(arr)

## Remarks
1. for cutting videos

        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        # ffmpeg_extract_subclip("full.mp4", start_seconds, end_seconds, targetname="cut.mp4")
        ffmpeg_extract_subclip("highway.mp4", 0, 10, targetname="cut.mp4")
