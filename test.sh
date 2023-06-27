mkdir -p output

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/defocus/merged/red-dahlia-flower-60597_2008_000902.png \
	--trimap-dir SIMD/defocus/trimap/red-dahlia-flower-60597_2008_000902.png \
	--output-dir output/defocus.png

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/fire/merged/b548f33ab9c559926d5f68088a00443a_2008_002601.png \
	--trimap-dir SIMD/fire/trimap/b548f33ab9c559926d5f68088a00443a_2008_002601.png \
	--output-dir output/fire.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/glass_ice/merged/clear-liquid-in-drinking-glass-1556381_2007_002260.png \
	--trimap-dir SIMD/glass_ice/trimap/clear-liquid-in-drinking-glass-1556381_2007_002260.png \
	--output-dir output/glass_ice.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/insect/merged/afeb913765053b516089cfcd645556b5--flying-insects-beautiful-bugs_2008_001712.png \
	--trimap-dir SIMD/insect/trimap/afeb913765053b516089cfcd645556b5--flying-insects-beautiful-bugs_2008_001712.png \
	--output-dir output/insect.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/motion/merged/creative+dance+headshots+minneapolis-3_2008_001140.png \
	--trimap-dir SIMD/motion/trimap/creative+dance+headshots+minneapolis-3_2008_001140.png \
	--output-dir output/motion.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/net/merged/HTB1ELiPXPfguuRjSspaq6yXVXXa6_2008_000727.png \
	--trimap-dir SIMD/net/trimap/HTB1ELiPXPfguuRjSspaq6yXVXXa6_2008_000727.png \
	--output-dir output/net.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/plastic_bag/merged/3_2007_003957.png \
	--trimap-dir SIMD/plastic_bag/trimap/3_2007_003957.png \
	--output-dir output/plastic_bag.png

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/sharp/merged/beautiful-bloom-blooming-blur-573020_2007_000837.png \
	--trimap-dir SIMD/sharp/trimap/beautiful-bloom-blooming-blur-573020_2007_000837.png \
	--output-dir output/sharp.png


python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/smoke_cloud/merged/59127bd249542_2008_003884.png \
	--trimap-dir SIMD/smoke_cloud/trimap/59127bd249542_2008_003884.png \
	--output-dir output/smoke_cloud.png

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/spider_web/merged/418943ddce09fb6388a001b06b1a2d68_2008_004985.png \
	--trimap-dir SIMD/spider_web/trimap/418943ddce09fb6388a001b06b1a2d68_2008_004985.png \
	--output-dir output/spider_web.png

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/water_drop/merged/dew-drops-2776772_1920_2007_002462.png \
	--trimap-dir SIMD/water_drop/trimap/dew-drops-2776772_1920_2007_002462.png \
	--output-dir output/water_drop.png

python run_one_image.py --model "vitmatte-s" --checkpoint-dir "checkpoints/ViTMatte_S_DIS.pth" \
	--image-dir SIMD/water_spray/merged/alcohol-alcoholic-beverage-cold-339696_2008_000950.png \
	--trimap-dir SIMD/water_spray/trimap/alcohol-alcoholic-beverage-cold-339696_2008_000950.png \
	--output-dir output/water_spray.png

