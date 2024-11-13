# ML-for-Robotics-Assignment-Code
Source Code for Project Implemented under ML for Robotics Course


**Abstract**



EVA (Environment Visual Assistant) is a robotic platform designed to enhance the independence and mobility of visually impaired individuals. By integrating computer vision, voice interaction, and obstacle avoidance, EVA provides real-time environmental information and personalized assistance. The system leverages a Raspberry Pi for processing, coupled with a camera, ultrasonic sensors, and Bluetooth communication with user-worn earbuds.
EVA operates in a continuous following mode, maintaining a safe distance behind the user through a human detection and tracking algorithm based on TensorFlow Lite. Two ultrasonic sensors enable ground obstacle detection, triggering avoidance maneuvers to ensure user safety. Voice interaction is initiated through a wake word detection system. Silero VAD constantly monitors streamed audio, activating openWakeWord upon detecting voice activity. Upon recognizing the wake word "Hey Eva," the Visual Assist pipeline is engaged.
This pipeline addresses user queries about the surrounding environment. Whisper ASR transcribes spoken commands into text, which is then processed alongside the current camera frame by the Qwen-2.5 VL visual language model. This model provides contextually relevant answers, which are converted to speech using a TTS engine and relayed back to the user through the earbuds.
This multi-modal approach enables EVA to provide a richer understanding of the environment compared to traditional assistive tools. The system's ability to follow, respond to voice commands, and answer visual questions has the potential to significantly improve the navigation experience for visually impaired individuals. Future work will focus on refining the obstacle avoidance capabilities, enhancing user interaction through natural language understanding, and personalizing models for individual user needs. EVA represents a promising step toward more intelligent and adaptive assistive technology for the visually impaired.



<img width="3196" alt="MLR" src="https://github.com/user-attachments/assets/726cc216-e5c9-4906-8b55-e2af2f0aa100">
