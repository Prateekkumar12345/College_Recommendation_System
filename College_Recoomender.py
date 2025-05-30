!pip install gradio groq pydantic pandas

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import gradio as gr
from groq import Groq
from pydantic import BaseModel, Field, validator
import getpass
import re
from pathlib import Path
import tempfile

# Define StudentData class with proper validation
class StudentData(BaseModel):
    Marks_10th: Optional[float] = Field(None, ge=0, le=100, description="10th standard marks percentage")
    Marks_12th: Optional[float] = Field(None, ge=0, le=100, description="12th standard marks percentage")
    JEE_Score: Optional[int] = Field(None, ge=1, description="JEE score/rank")
    Budget: Optional[int] = Field(None, ge=0, description="Budget for education in rupees")
    Preferred_Location: Optional[str] = Field(None, description="Preferred study location")
    Gender: Optional[str] = Field(None, description="Student's gender")
    Target_Exam: Optional[str] = Field(None, description="Target entrance exams")
    State_Board: Optional[str] = Field(None, description="Educational board")
    Category: Optional[str] = Field(None, description="Reservation category")
    Extra_Curriculars: Optional[str] = Field(None, description="Extracurricular activities")
    Future_Goal: Optional[str] = Field(None, description="Career goals")

    @validator('Budget', pre=True)
    def convert_budget(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Convert lakhs and crores to actual numbers
            v = v.lower().replace(',', '').replace(' ', '')
            if 'lakh' in v or 'lac' in v:
                num = re.findall(r'\d+\.?\d*', v)
                if num:
                    return int(float(num[0]) * 100000)
            elif 'crore' in v:
                num = re.findall(r'\d+\.?\d*', v)
                if num:
                    return int(float(num[0]) * 10000000)
            else:
                # Try to extract number
                num = re.findall(r'\d+', v)
                if num:
                    return int(num[0])
        return v

    @validator('Gender', pre=True)
    def normalize_gender(cls, v):
        if v is None:
            return None
        v = str(v).lower()
        if v in ['male', 'boy', 'm', 'man']:
            return 'Male'
        elif v in ['female', 'girl', 'f', 'woman']:
            return 'Female'
        else:
            return v.title()

class CollegeCounselorChatbot:
    def __init__(self, api_key, name="Lauren"):
        self.name = name
        self.model = "llama-3.1-8b-instant"
        self.client = Groq(api_key=api_key)

        # Initialize student data object
        self.student_data = StudentData()
        self.data_collected = False
        self.recommendations_provided = False
        self.profile_filename = None

        # Create profiles directory and initialize profile file
        self.profiles_dir = self.create_profiles_directory()
        self.initialize_profile_file()

        self.conversation_history = [
            {"role": "system", "content": f"""
            You are {name}, an AI college counselor for Indian students. Your goal is to collect information about the student and provide personalized college recommendations.

            You need to gather these key details from the student through natural conversation:
            1. Marks_10th - Student's 10th standard marks percentage
            2. Marks_12th - Student's 12th standard marks percentage
            3. JEE_Score - JEE score if applicable
            4. Budget - How much they can afford for their entire education
            5. Preferred_Location - Which part of India they prefer to study in
            6. Gender - Student's gender
            7. Target_Exam - Which entrance exams they're targeting
            8. State_Board - Which educational board they studied under
            9. Category - Their reservation category (General, OBC, SC, ST, etc.)
            10. Extra_Curriculars - Any extracurricular activities/achievements
            11. Future_Goal - Career aspirations or goals

            Be friendly, conversational, and encouraging. First introduce yourself briefly and start collecting information.
            Only move on to college recommendations after collecting all the necessary information.
            """}
        ]

        # Sample college database
        self.colleges = [
            {"name": "IIT Bombay", "min_jee": 8000, "fees": 800000, "location": "Mumbai", "acceptance_rate": "Very Low", "specialties": ["Engineering", "Technology"]},
            {"name": "IIT Delhi", "min_jee": 9000, "fees": 750000, "location": "Delhi", "acceptance_rate": "Very Low", "specialties": ["Engineering", "Computer Science"]},
            {"name": "BITS Pilani", "min_jee": 15000, "fees": 1200000, "location": "Rajasthan", "acceptance_rate": "Low", "specialties": ["Engineering", "Pharmacy"]},
            {"name": "VIT Vellore", "min_jee": 50000, "fees": 900000, "location": "Tamil Nadu", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Bio-Technology"]},
            {"name": "Manipal Institute of Technology", "min_jee": 70000, "fees": 1500000, "location": "Karnataka", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Medicine"]},
            {"name": "NIT Trichy", "min_jee": 20000, "fees": 500000, "location": "Tamil Nadu", "acceptance_rate": "Low", "specialties": ["Engineering"]},
            {"name": "Delhi University", "min_jee": None, "fees": 200000, "location": "Delhi", "acceptance_rate": "Moderate", "specialties": ["Arts", "Commerce", "Science"]},
            {"name": "AIIMS Delhi", "min_jee": None, "fees": 600000, "location": "Delhi", "acceptance_rate": "Very Low", "specialties": ["Medicine"]},
            {"name": "Tula's Institute", "min_jee": 100000, "fees": 600000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["BCA", "MCA", "BBA", "MBA"]},
            {"name": "Graphic Era University", "min_jee": 100000, "fees": 700000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Management", "Computer Science"]},
            {"name": "Doon University", "min_jee": 100000, "fees": 400000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["Science", "Arts", "Commerce"]},
        ]

    def create_profiles_directory(self):
        """Create profiles directory if it doesn't exist"""
        try:
            # Try current directory first
            profiles_dir = Path('./student_profiles')
            profiles_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = profiles_dir / 'test_write.txt'
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()  # Delete test file

            print(f"✅ Profiles directory created/verified at: {profiles_dir.absolute()}")
            return profiles_dir

        except Exception as e:
            print(f"❌ Error with ./student_profiles directory: {e}")
            try:
                # Fallback to temp directory
                import tempfile
                profiles_dir = Path(tempfile.gettempdir()) / 'student_profiles'
                profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using temporary directory: {profiles_dir.absolute()}")
                return profiles_dir
            except Exception as e2:
                print(f"❌ Error creating temp directory: {e2}")
                # Last resort - current directory
                return Path('.')

    def initialize_profile_file(self):
        """Initialize the profile file immediately"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.profile_filename = self.profiles_dir / f"student_profile_{timestamp}.txt"

            # Create initial empty profile
            initial_profile = {
                "student_profile": {
                    "session_info": {
                        "profile_created": datetime.now().isoformat(),
                        "counselor_name": self.name,
                        "session_id": timestamp,
                        "data_completion_status": "In Progress",
                        "last_updated": datetime.now().isoformat()
                    },
                    "collected_data": {},
                    "missing_fields": list(self.student_data.__fields__.keys())
                }
            }

            # Write initial profile
            with open(self.profile_filename, 'w', encoding='utf-8') as f:
                json.dump(initial_profile, f, indent=4, ensure_ascii=False)

            print(f"✅ Profile file initialized: {self.profile_filename}")
            return True

        except Exception as e:
            print(f"❌ Error initializing profile file: {e}")
            return False

    def extract_information_with_llm(self, user_message):
        """Use LLM to extract structured information from user messages"""
        current_data = self.student_data.dict()

        extraction_prompt = f"""
        Extract student information from the message and return ONLY a valid JSON object.

        Current data: {json.dumps(current_data, default=str)}

        User message: "{user_message}"

        Extract any of these fields (only include if clearly mentioned):
        - Marks_10th: percentage (0-100)
        - Marks_12th: percentage (0-100)
        - JEE_Score: rank/score (positive integer)
        - Budget: amount in rupees (convert lakhs/crores: 5 lakhs = 500000)
        - Preferred_Location: city/state in India
        - Gender: Male/Female/Other
        - Target_Exam: JEE/NEET/etc
        - State_Board: CBSE/ICSE/State Board/etc
        - Category: General/OBC/SC/ST/EWS/etc
        - Extra_Curriculars: activities/achievements
        - Future_Goal: career aspirations

        Return ONLY valid JSON. If nothing found, return {{}}.

        Examples:
        "I got 85% in 10th" → {{"Marks_10th": 85}}
        "Budget is 5 lakhs" → {{"Budget": 500000}}
        "I'm a boy from Delhi" → {{"Gender": "Male", "Preferred_Location": "Delhi"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            extracted_text = response.choices[0].message.content.strip()
            print(f"🔍 LLM Extraction Response: {extracted_text}")

            # Parse JSON from response
            try:
                json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    extracted_data = json.loads(json_str)

                    if extracted_data:  # Only update if we extracted something
                        # Update current data with new values
                        for key, value in extracted_data.items():
                            if value is not None and value != "":
                                current_data[key] = value

                        # Validate with Pydantic
                        self.student_data = StudentData(**current_data)
                        print(f"📊 Updated student data: {self.student_data.dict()}")

                        # Save profile after each update
                        self.save_profile_json()
                        return True

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
            return False

        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            return False

    def check_data_completion(self):
        """Check if all required data has been collected"""
        data_dict = self.student_data.dict()
        missing_fields = [k for k, v in data_dict.items() if v is None]
        return len(missing_fields) == 0

    def get_missing_fields(self):
        """Get list of missing fields"""
        data_dict = self.student_data.dict()
        return [k for k, v in data_dict.items() if v is None]

    def get_next_question(self):
        """Generate smart follow-up questions based on missing data"""
        missing = self.get_missing_fields()
        if not missing:
            return None

        question_map = {
            'Marks_10th': "Could you please share your 10th standard marks percentage?",
            'Marks_12th': "What were your 12th standard marks percentage?",
            'JEE_Score': "Did you take JEE? If so, what was your score or rank?",
            'Budget': "What's your budget for the entire course (you can mention in lakhs)?",
            'Preferred_Location': "Which part of India would you prefer to study in?",
            'Gender': "Just for better recommendations, could you let me know your gender?",
            'Target_Exam': "Which entrance exams are you preparing for or have taken?",
            'State_Board': "Which educational board did you study under (CBSE, ICSE, State Board)?",
            'Category': "What's your reservation category (General, OBC, SC, ST, etc.)?",
            'Extra_Curriculars': "Do you have any extracurricular activities or achievements?",
            'Future_Goal': "What are your career goals or what field interests you?"
        }

        return question_map.get(missing[0], f"Could you tell me about your {missing[0]}?")

    def generate_comprehensive_profile_json(self):
        """Generate comprehensive student profile with all details"""
        data_dict = self.student_data.dict()
        timestamp = datetime.now()

        # Get collected vs missing data
        collected_data = {k: v for k, v in data_dict.items() if v is not None}
        missing_fields = [k for k, v in data_dict.items() if v is None]

        profile_data = {
            "student_profile": {
                "session_info": {
                    "profile_created": timestamp.isoformat(),
                    "counselor_name": self.name,
                    "session_id": timestamp.strftime("%Y%m%d_%H%M%S"),
                    "data_completion_status": "Complete" if self.data_collected else "In Progress",
                    "missing_fields": missing_fields,
                    "collected_fields": list(collected_data.keys()),
                    "completion_percentage": f"{((11 - len(missing_fields)) / 11) * 100:.1f}%",
                    "last_updated": timestamp.isoformat()
                },
                "personal_information": {
                    "gender": data_dict.get('Gender'),
                    "category": data_dict.get('Category'),
                    "preferred_location": data_dict.get('Preferred_Location')
                },
                "academic_details": {
                    "marks_10th_percentage": data_dict.get('Marks_10th'),
                    "marks_12th_percentage": data_dict.get('Marks_12th'),
                    "jee_score_rank": data_dict.get('JEE_Score'),
                    "state_board": data_dict.get('State_Board'),
                    "target_examinations": data_dict.get('Target_Exam')
                },
                "preferences_and_goals": {
                    "education_budget_inr": data_dict.get('Budget'),
                    "career_goals": data_dict.get('Future_Goal'),
                    "extracurricular_activities": data_dict.get('Extra_Curriculars')
                },
                "raw_collected_data": collected_data
            }
        }

        # Add college recommendations if data is complete
        if self.data_collected:
            recommendations = self.get_college_recommendations()
            profile_data["college_recommendations"] = {
                "recommendation_date": timestamp.isoformat(),
                "total_recommendations": len(recommendations),
                "colleges": recommendations
            }

        return profile_data

    def get_college_recommendations(self):
        """Get filtered college recommendations based on student profile"""
        recommendations = []
        data_dict = self.student_data.dict()

        for college in self.colleges:
            match_score = 0
            match_reasons = []

            # Location matching
            if (data_dict.get('Preferred_Location') and
                data_dict['Preferred_Location'].lower() in college['location'].lower()):
                match_score += 2
                match_reasons.append(f"Located in preferred area: {college['location']}")

            # Budget matching
            if data_dict.get('Budget') and college['fees'] <= data_dict['Budget']:
                match_score += 2
                match_reasons.append(f"Fits within budget (₹{college['fees']:,})")

            # JEE Score matching
            if (data_dict.get('JEE_Score') and college.get('min_jee') and
                data_dict['JEE_Score'] <= college['min_jee']):
                match_score += 3
                match_reasons.append(f"JEE rank qualifies (Min required: {college['min_jee']})")

            # Add basic matches for all colleges
            if match_score == 0:
                match_score = 1
                match_reasons.append("General recommendation based on profile")

            recommendations.append({
                "college_name": college['name'],
                "location": college['location'],
                "annual_fees_inr": college['fees'],
                "specialties": college['specialties'],
                "acceptance_rate": college['acceptance_rate'],
                "match_score": match_score,
                "match_reasons": match_reasons,
                "minimum_jee_rank_required": college.get('min_jee')
            })

        # Sort by match score and return top 5
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:5]

    def save_profile_json(self):
        """Save student profile to JSON format in a text file"""
        try:
            profile_data = self.generate_comprehensive_profile_json()

            # Ensure we have a filename
            if not self.profile_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.profile_filename = self.profiles_dir / f"student_profile_{timestamp}.txt"

            # Create directory if it doesn't exist
            self.profile_filename.parent.mkdir(parents=True, exist_ok=True)

            # Write with error handling
            with open(self.profile_filename, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            print(f"✅ Profile saved to: {self.profile_filename}")
            return str(self.profile_filename)

        except Exception as e:
            print(f"❌ Error saving profile: {e}")
            return None

    def get_profile_content_for_download(self):
        """Get profile content formatted for download"""
        try:
            profile_data = self.generate_comprehensive_profile_json()
            return json.dumps(profile_data, indent=4, ensure_ascii=False, sort_keys=True)
        except Exception as e:
            print(f"❌ Error generating profile content: {e}")
            return json.dumps({"error": f"Could not generate profile: {str(e)}"}, indent=4)

    def chat(self, message, history):
        """Process user message and generate response"""
        print(f"💬 Processing message: {message[:50]}...")

        # Extract information from user message
        extraction_success = self.extract_information_with_llm(message)

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Check if we have all required information
        if not self.data_collected:
            self.data_collected = self.check_data_completion()
            if self.data_collected:
                print("🎉 All required data collected!")
                self.save_profile_json()  # Save complete profile

                student_profile = "\n".join([f"{k}: {v}" for k, v in self.student_data.dict().items()])
                colleges_data = self.format_colleges_for_prompt()

                self.conversation_history.append({
                    "role": "system",
                    "content": f"""
                    PROVIDE RECOMMENDATIONS NOW. All required information has been collected.

                    Student Profile:
                    {student_profile}

                    {colleges_data}

                    Based on this student's profile, recommend 3-5 suitable colleges that match their profile.
                    For each recommendation, explain:
                    1. Why this college is a good fit
                    2. Key programs relevant to their interests
                    3. Admission requirements and competitiveness
                    4. Estimated costs and how it fits their budget

                    Also provide practical next steps for applications.
                    End by mentioning they can download their complete profile using the download button.
                    """
                })

                self.recommendations_provided = True

            else:
                # Continue collecting information - save partial profile
                self.save_profile_json()  # Save after every update

                missing_fields = self.get_missing_fields()
                next_question = self.get_next_question()

                collected_info = {k: v for k, v in self.student_data.dict().items() if v is not None}

                self.conversation_history.append({
                    "role": "system",
                    "content": f"""
                    Continue collecting information. Acknowledge what they shared and ask: "{next_question}"

                    Information collected so far: {json.dumps(collected_info, default=str)}
                    Still missing: {missing_fields}

                    Be encouraging and conversational. Mention that their profile is being saved automatically.
                    """
                })

        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history[-10:],  # Keep recent context
                temperature=0.7,
                max_tokens=1000,
            )

            assistant_response = response.choices[0].message.content

        except Exception as e:
            assistant_response = f"I'm sorry, there was an error: {str(e)}"
            print(f"❌ Chat Error: {e}")

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response

    def format_colleges_for_prompt(self):
        """Format college data for inclusion in the prompt"""
        colleges_text = "College Database:\n"
        for college in self.colleges:
            colleges_text += f"- {college['name']}: Location: {college['location']}, "
            colleges_text += f"Min JEE Rank (if applicable): {college['min_jee']}, "
            colleges_text += f"Approximate Fees: {college['fees']}, "
            colleges_text += f"Acceptance Rate: {college['acceptance_rate']}, "
            colleges_text += f"Specialties: {', '.join(college['specialties'])}\n"
        return colleges_text

def create_chatbot_interface(api_key):
    """Create the Gradio interface"""
    counselor = CollegeCounselorChatbot(api_key=api_key, name="Lauren")

    with gr.Blocks(title="Lauren - AI College Counselor") as app:
        gr.Markdown("# 🎓 Lauren - Your AI College Counselor")
        gr.Markdown(f"""
        Welcome! I'll help you find the right colleges based on your profile.
        Share your academic details, preferences, and goals through natural conversation.

        📁 **Your profile will be automatically saved to:** `{counselor.profiles_dir.absolute()}`
        📝 **Profile file:** `{counselor.profile_filename.name if counselor.profile_filename else 'Not yet created'}`
        """)

        chatbot = gr.Chatbot(height=500, show_copy_button=True)

        msg = gr.Textbox(
            placeholder="Hi Lauren! I need help finding the right college...",
            container=False,
            scale=7
        )

        with gr.Row():
            submit = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear", scale=1)

        with gr.Row():
            download_btn = gr.Button("📄 Download Profile JSON", variant="secondary")
            download_file = gr.File(label="Download Your Profile", visible=False)

        # Status display
        status_display = gr.Markdown("🔄 **Status:** Ready to start collecting your information...")

        # Initial greeting
        initial_greeting = f"""
        👋 Hi! I'm Lauren, your AI college counselor.

        I'll help you find the best colleges for your profile and goals.
        Let's start with your academic background - could you tell me your 10th and 12th standard marks?

        📁 **Note:** Your profile is being saved automatically to: `{counselor.profile_filename}`

        I'll collect information about:
        • Academic marks (10th, 12th)
        • Entrance exam scores (JEE, etc.)
        • Budget and preferences
        • Career goals
        """

        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history, gr.update(), gr.update(visible=False)

            response = counselor.chat(message, chat_history)
            chat_history.append((message, response))

            # Update status
            collected_fields = [k for k, v in counselor.student_data.dict().items() if v is not None]
            total_fields = len(counselor.student_data.__fields__)
            completion_pct = (len(collected_fields) / total_fields) * 100

            status_text = f"""
            📊 **Profile Status:** {completion_pct:.1f}% complete ({len(collected_fields)}/{total_fields} fields)
            📁 **File:** `{counselor.profile_filename.name if counselor.profile_filename else 'Not created'}`
            ✅ **Collected:** {', '.join(collected_fields) if collected_fields else 'None yet'}
            """

            return "", chat_history, gr.update(value=status_text), gr.update(visible=False)

        def clear_conversation():
            nonlocal counselor
            counselor = CollegeCounselorChatbot(api_key=api_key, name="Lauren")
            status_text = f"""
            🔄 **Status:** New session started
            📁 **File:** `{counselor.profile_filename.name if counselor.profile_filename else 'Not created'}`
            """
            return [], gr.update(value=status_text), gr.update(visible=False)

        def download_profile():
            try:
                # Always try to generate downloadable content
                content = counselor.get_profile_content_for_download()

                if not content or content.strip() == "":
                    return gr.update(visible=False)

                # Create a temporary file for download
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_student_profile_{timestamp}.txt',
                                               delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                print(f"📁 Download file created: {temp_file_path}")
                return gr.update(value=temp_file_path, visible=True)

            except Exception as e:
                print(f"❌ Download error: {e}")
                return gr.update(visible=False)

        # Event handlers
        msg.submit(respond, [msg, chatbot], [msg, chatbot, status_display, download_file])
        submit.click(respond, [msg, chatbot], [msg, chatbot, status_display, download_file])
        clear.click(clear_conversation, None, [chatbot, status_display, download_file])
        download_btn.click(download_profile, None, [download_file])

        # Set initial greeting
        chatbot.value = [("", initial_greeting)]

    return app

def main():
    print("🚀 Starting College Counselor Chatbot...")

    api_key = getpass.getpass("Enter your Groq API Key: ")

    if not api_key:
        print("❌ No API key provided.")
        return

    print("✅ API key received. Creating interface...")

    app = create_chatbot_interface(api_key)

    print("🌐 Launching application...")
    app.launch(share=True)

if __name__ == "__main__":
    main()