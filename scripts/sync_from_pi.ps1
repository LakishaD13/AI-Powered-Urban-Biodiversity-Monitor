$PI_USER = "lakishaPi"
$PI_IP   = "172.17.0.1"   

$REMOTE_IMG_DIR = "/home/pi/bird_project/capture/pir_capture/"
$LOCAL_IMG_DIR  = "C:/Users/laksh/OneDrive/Documents/bird_project/pi_imports/images/"

New-Item -ItemType Directory -Force -Path $LOCAL_IMG_DIR | Out-Null

# Use scp to copy ONLY images that don't already exist locally
# - This is simple + reliable; it won't overwrite same-name files.
scp "$PI_USER@$PI_IP`:$REMOTE_IMG_DIR*.jpg" $LOCAL_IMG_DIR
scp "$PI_USER@$PI_IP`:$REMOTE_IMG_DIR*.png" $LOCAL_IMG_DIR
scp "$PI_USER@$PI_IP`:$REMOTE_IMG_DIR*.jpeg" $LOCAL_IMG_DIR