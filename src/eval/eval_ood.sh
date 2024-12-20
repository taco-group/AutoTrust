# CUDA_VISIBLE_DEVICES='1' python main.py \
#     --model-path "/home/xingshuo/Reward-Model/llava-v1.6-mistral-7b" \
#     --image-folder "/home/xingshuo/Trustworthy-DriveVLM/dataset/DriveLM" \
#     --question-file "/home/xingshuo/Trustworthy-DriveVLM/dataset/DriveLM/quetions_with_type.json" \
#     --answers-file "/home/xingshuo/Trustworthy-DriveVLM/dataset/DriveLM/answer.json" 




# question_file='/home/shuoxing/Trustworthy-DriveVLM/dataset/DriveLM/raw/v1_1_val_nus_q_only_filtered_gpt4o_copy.json'
# image_folder='/home/shuoxing/Trustworthy-DriveVLM/OOD_dataset/Noisy_data/DriveLM_Noise'


# key='dolphins_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_dolphins.json'
# output_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/evaluation_file/language/DriveLM_Noise_answer_dolphins.json'

# echo "Running evaluation on dataset: $predictions_file"


# python3  /home/jinlong/Trustworthy-DriveVLM/src/eval/eval_gpt_score_ood.py \
#     --questions_file $question_file \
#     --predictions_file $predictions_file \
#     --output_file  $output_file \
#     --image_folder  $image_folder\
#     --ans_key $key




# ans_key='dolphins_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_dolphins.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_dolphins.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_Dolphins_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_Dolphins_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_Dolphins_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_Dolphins_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_Dolphins_Spanish.json'

# ans_key='llava_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_llava.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_llava.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_llava_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_llava_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_llava_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_llava_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_llava_Spanish.json'




ans_key='gpt_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_gpt.json'
predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_gpt.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_gpt_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_gpt_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_gpt_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_gpt_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_gpt_Spanish.json'




# ans_key='drivelm_agent_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_drivelm_agent.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_drivelm_agent.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm-agent_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm-agent_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm-agent_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm-agent_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm-agent_Spanish.json'




# ans_key='llama_adapter_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_drivelm_challenge.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_drivelm_challenge.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm_challenge_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm_challenge_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm_challenge_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm_challenge_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_drivelm_challenge_Spanish.json'


# ans_key='EM_VLM4AD_ood_answer'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/NuScenes_Noise_answer_EM_VLM4AD.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/DriveLM_robustness_answer_EM_VLM4AD.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_EM_VLM4AD_english.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_EM_VLM4AD_chinese.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_EM_VLM4AD_Arabic.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_EM_VLM4AD_Hindi.json'
# predictions_file='/home/shuoxing/Trustworthy-DriveVLM/Robustness/answers/robustness/language/DriveLM_robustness_answer_EM_VLM4AD_Spanish.json'



python3  /home/shuoxing/Trustworthy-DriveVLM/src/eval/eval_close_end.py \
    --predictions_file $predictions_file \
    --ans_key $ans_key \


# python3 /home/shuoxing/Trustworthy-DriveVLM/src/eval/eval_abs.py \
#     --test_data_path $predictions_file \
#     --ans_key $ans_key \