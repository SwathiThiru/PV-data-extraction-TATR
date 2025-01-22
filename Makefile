training-detection:
	clear
	python src/main.py --data_type detection --config_file detection_config.json --data_root_dir ../../PubTablesDataset/PubTables-1M-Detection

training-recognition:
	clear
	python src/main.py --data_type structure --config_file structure_config.json --device cpu --data_root_dir ../../PubTablesDataset/PubTables-1M-Structure

inference-detection:
	clear
	python src/inference.py --mode detect --detection_config_path detection_config.json --detection_model_path ../weights/pubtables1m_detection_detr_r18.pth --detection_device cpu --in_dir ../inferences/pdf  --out_dir ../inferences/detectionOutput --crop_padding 5 -p

inference-regonition:
	clear
	python src/inference.py --mode recognize --structure_config_path structure_config.json --structure_model_path ../weights/pubtables1m_structure_detr_r18.pth --structure_device cpu --in_dir ../inferences/detectionOutput  --out_dir  ../inferences/recognitionOutput -c -z

inference-extraction:
	clear
	python src/inference.py --mode extract --detection_config_path detection_config.json --detection_model_path ../weights/pubtables1m_detection_detr_r18.pth --detection_device cpu --structure_config_path structure_config.json --structure_model_path ../weights/pubtables1m_structure_detr_r18.pth --structure_device cpu --in_dir ../inferences/pdf --out_dir ../inferences/extractionOutput -c -z --crop_padding 20

evaluation-detection-cells:
	clear
	python src/TATR_detection_evaluation.py --data_type detection --config_file detection_config.json --data_root_dir ../../TATREvaluationDataset/solarCellDetection --model_load_path ../weights/pubtables1m_detection_detr_r18.pth --device cpu

evaluation-detection-modules:
	clear
	python src/TATR_detection_evaluation.py --data_type detection --config_file detection_config.json --data_root_dir ../../TATREvaluationDataset/solarModuleDetection --model_load_path ../weights/pubtables1m_detection_detr_r18.pth --device cpu

evaluation-extraction-cells:
	clear
	

evaluation-extraction-module:
	clear