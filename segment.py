import time
start_time = time.time()
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image
import numpy as np

'''prompt = "woman"

model = LangSAM()
print(model)
image_pil = Image.open("./images/park.png").convert("RGB")

masks, boxes, phrases, logits = model.predict(image_pil, prompt)

masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
final_mask = np.sum(masks_np, axis=0)
final_mask[final_mask == 1] = 255

mask_image = Image.fromarray(final_mask).convert("RGB")
# mask_image.show()

# exit()
# final_mask = masks_np[0].copy()
# for mask in masks_np[1:]:
#     final_mask[mask == 255] = 255

# mask_image = Image.fromarray(masks_np[0]).convert("RGB")
# mask_image = Image.fromarray(final_mask).convert("RGB")
# mask_image.show()
mask_image.save("./images/mask.png")

print("--- %s seconds ---" % (time.time() - start_time))
# Approx time: 4.75 mins'''


# working

import time
import numpy as np
from PIL import Image

# Add the provided code snippet here

prompt = "woman"

model = LangSAM()
print(model)
image_pil = Image.open("park.png").convert("RGB")

start_time = time.time()

masks, boxes, phrases, logits = model.predict(image_pil, prompt)

masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
final_mask = np.sum(masks_np, axis=0)
#final_mask[final_mask == 1] = 255

# Ensure that final_mask is a valid NumPy array representing image data
final_mask = final_mask.astype(np.uint8)  # Convert data type to uint8 if necessary

# If necessary, adjust the values in final_mask to ensure they are within the valid range for image pixel values
final_mask[final_mask == 1] = 255  # Adjust values if needed

# Create a PIL Image from the final_mask array
mask_image = Image.fromarray(final_mask)

# Convert to "RGB" mode if necessary
mask_image = mask_image.convert("RGB")
mask_image.show()

# Save or display the image as needed
#mask_image.save("./images/mask.png")

# Print the time taken to execute the code
print("--- %s seconds ---" % (time.time() - start_time))
