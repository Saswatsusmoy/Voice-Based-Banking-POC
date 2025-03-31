# Voice-Driven Banking via LAMs

## Technical Architecture and Implementation

---

## Project Overview

This project demonstrates a voice-based banking platform that supports multiple languages including low-resource languages. It uses Large Acoustic Models (LAMs) to enable voice-controlled banking operations, making financial services more accessible to users who may face challenges with traditional text-based interfaces.

---

## System Architecture

The system follows a client-server architecture with:

- **Frontend**: HTML, CSS, and JavaScript providing a responsive web interface
- **Backend**: Flask-based Python server handling speech processing, intent recognition, and banking operations
- **External Services**: Utilizing Azure Speech Services and Google Speech APIs
- **Local Models**: Leveraging Wav2Vec2 models for offline speech recognition

![System Architecture](https://yuml.me/diagram/plain/class/[Client]->[Flask Server], [Flask Server]->[Speech Recognition], [Flask Server]->[Intent Recognition], [Flask Server]->[Voice Biometrics], [Flask Server]->[Banking Services], [Speech Recognition]->[Azure API], [Speech Recognition]->[Google API], [Speech Recognition]->[Local Models])

---

## Directory Structure

```
/
├── app.py                  # Main Flask application
├── /config/         
│   ├── intent_patterns.json # Language-specific patterns for intent recognition
│   └── azure_config.json    # Azure speech service configuration
├── /data/            
│   ├── mock_db.json        # Mock banking data
│   ├── users.json          # User data
│   └── /voice_prints/      # Voice authentication models
├── /models/          
│   ├── speech_recognition.py # Speech-to-text conversion
│   ├── intent_recognition.py # Banking intent detection
│   └── voice_biometrics.py   # Voice authentication logic
├── /services/        
│   ├── banking_service.py  # Banking operations
│   └── user_service.py     # User management
├── /static/          
│   ├── /css/
│   │   └── style.css       # Frontend styling
│   └── /js/
│       └── app.js          # Frontend logic
└── /templates/      
    └── index.html          # Main application page
```

---

## Core Components: Speech Recognition

The `speech_recognition.py` module provides a tiered approach to convert speech to text:

1. **Input Processing**:

   - Audio files are first converted to WAV format using `convert_audio_format()`
   - Multiple conversion methods are attempted (pydub, ffmpeg, librosa)
2. **Service Selection Logic**:

   - Language-specific routing to the most appropriate service
   - Fallback mechanisms if primary services fail
   - Configuration defined in `azure_config.json`
3. **Recognition Services**:

   - **Azure Speech Services** (`recognize_with_azure()`): Primary for non-English languages
   - **Google Speech Recognition** (`recognize_with_google()`): Optimized for English
   - **Local Wav2Vec2 Models** (`recognize_with_local_model()`): Offline fallback option
4. **Error Handling**:

   - Detailed logging of each step
   - Graceful degradation through fallback mechanisms
   - Transparent error messages for debugging

```python
def recognize_speech(audio_path, language='en-US'):
    try:
        wav_path = convert_audio_format(audio_path)
  
        # For English, use Google STT as primary service
        if language == 'en-US':
            result = recognize_with_google(wav_path, language)
            if result:
                return result
            # Fallback to Azure if Google fails
            result = recognize_with_azure(wav_path, language)
            return result if result else "Speech recognition could not understand audio"
  
        # For other languages, follow the fallback priority chain
        errors = []
        for service in FALLBACK_PRIORITY:  # ["azure", "google", "local"]
            if service == "azure":
                result = recognize_with_azure(wav_path, language)
                if result: return result
            elif service == "google":
                result = recognize_with_google(wav_path, language)
                if result: return result
            elif service == "local":
                result = recognize_with_local_model(wav_path, language)
                if result: return result
  
        # If all methods failed
        return "Speech recognition could not understand audio"
    except Exception as e:
        return f"Error processing speech: {str(e)}"
```

---

## Core Components: Intent Recognition

The `intent_recognition.py` module identifies what banking operation the user wants to perform:

1. **Pattern Matching**:

   - Language-specific regex patterns from `intent_patterns.json`
   - Patterns for common banking intents (balance check, transfers, transaction history)
2. **Parameter Extraction**:

   - Language-specific extractors for transaction amounts and recipients
   - `extract_transfer_params_english()`, `extract_transfer_params_hindi()`, etc.
3. **Fallback Mechanism**:

   - Keyword-based intent detection when pattern matching fails
   - Language-specific keywords for each intent type
4. **NLP Processing**:

   - Using spaCy models for text processing (`load_nlp_model()`)
   - Language-specific models loaded based on the detected language
5. **Semantic Matching**:

   - Utilizes pre-trained language models for semantic similarity
   - Matches user input to intent patterns based on meaning, not just keywords

```python
def extract_intent(text, language='en-US'):
    # Normalize text
    normalized_text = ' '.join(text.lower().split())
  
    # Try pattern matching first (most accurate)
    language_patterns = INTENT_PATTERNS.get(language, INTENT_PATTERNS.get('en-US'))
    for intent, patterns in language_patterns.items():
        for pattern in patterns['patterns']:

            if re.search(pattern, normalized_text):
                intent_data = {'intent_type': intent, 'parameters': {}}
          
                # Extract parameters based on intent type and language
                if intent == 'transfer_money':
                    if language == 'hi-IN':
                        intent_data['parameters'] = extract_transfer_params_hindi(normalized_text)
                    elif language == 'sw':
                        intent_data['parameters'] = extract_transfer_params_swahili(normalized_text)
                    else:  # Default to English
                        intent_data['parameters'] = extract_transfer_params_english(normalized_text)
                  
                # Similar extraction for other intent types...
                return intent_data
  
    # If no pattern matched, try keyword matching
    # [Keyword matching implementation]
  
    return {'intent_type': 'unknown', 'parameters': {}}
```

---

## Core Components: Voice Biometrics

The `voice_biometrics.py` module provides voice-based authentication:

1. **Feature Extraction**:

   - Extracts MFCC (Mel-Frequency Cepstral Coefficients) from audio samples
   - Normalizes features for consistent comparison
2. **Voice Print Creation**:

   - Trains a Gaussian Mixture Model (GMM) on user's voice features
   - Stores serialized model in `voice_prints_dir` for future authentication
3. **Authentication Process**:

   - Extracts features from new audio sample
   - Compares against stored voice print using log-likelihood score
   - Implements adaptive thresholding for authentication decisions

```python
def extract_voice_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
  
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  
    # Normalize features
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
  
    return mfccs.T  # Transpose for sklearn compatibility

def authenticate_voice(audio_path, user_id, threshold=None):
    # Load the user's voice model
    with open(get_voice_print_path(user_id), 'rb') as f:
        gmm = pickle.load(f)
  
    # Extract features from the provided audio
    features = extract_voice_features(audio_path)
  
    # Calculate log likelihood
    score = gmm.score(features)
  
    # Use adaptive thresholding
    if threshold is None:
        threshold = -80  # Default value based on typical log-likelihood scores
  
    authenticated = score > threshold
  
    return {
        'authenticated': authenticated,
        'confidence': score,
        'threshold': threshold,
        'user_id': user_id
    }
```

---

## Core Components: Banking Services

The `banking_service.py` module simulates banking operations:

1. **Mock Database**:

   - JSON-based storage in `mock_db.json`
   - User accounts, balances, and transaction history
2. **Banking Operations**:

   - **Balance Check**: Retrieves account balances
   - **Money Transfers**: Updates balances and creates transaction records for both parties
   - **Transaction History**: Filters and returns recent transactions
3. **Data Generation**:

   - Creates realistic mock data for demonstration
   - Maintains transaction records with timestamps and descriptions

```python
def process_banking_request(intent_data, user):
    intent_type = intent_data['intent_type']
    parameters = intent_data['parameters']
  
    # Load the mock database
    db = load_mock_db()
    user_data = db['users'].get(user['id'])
  
    if intent_type == 'check_balance':
        response = {
            'success': True,
            'intent_type': intent_type,
            'accounts': user_data['accounts'],
            'message': "Your current balances are: ..."
        }
  
    elif intent_type == 'transfer_money':
        # Implement transfer logic, update balances, add transaction records
        # [Transfer implementation]
  
    elif intent_type == 'transaction_history':
        # Filter and return transaction history
        # [Transaction history implementation]
  
    return response
```

---

## Core Components: User Services

The `user_service.py` module manages user accounts:

1. **User Management**:

   - User creation with `create_user()`
   - Authentication with `authenticate_user()`
   - Secure password handling with `werkzeug.security`
2. **Data Storage**:

   - User profiles stored in `users.json`
   - Password hashing for security
3. **Language Preferences**:

   - Each user has a language preference
   - Updates via `update_user_language()`

```python
def authenticate_user(username, password):
    user = get_user_by_username(username)
    if not user:
        return None
  
    if check_password_hash(user['password_hash'], password):
        return user
  
    return None

def create_user(username, password, name, email, phone, language='en-US'):
    # Check if username exists
    # Create new user with hashed password
    # Save to users database
    return {'success': True, 'user_id': new_user_id}
```

---

## API Routes and Request Flow

The `app.py` file defines the Flask application with these key routes:

1. **'/api/process-voice'** (POST):

   - Receives audio file and user ID
   - Authenticates the user's voice
   - Processes speech to text
   - Extracts intent
   - Executes banking operation
   - Returns results to the client
2. **Authentication Routes**:

   - '/api/login' - User authentication
   - '/api/register' - New user registration
   - '/api/update-language' - Update user language preference
3. **Voice Enrollment**:

   - '/api/enroll-voice' - Creates voice print for new users
4. **Health Check**:

   - '/api/health' - Monitoring system status

---

## User Experience Flow

The frontend (defined in `index.html` and `app.js`) provides:

1. **Authentication Interface**:

   - Login form
   - Registration form
   - Language selection
2. **Voice Banking Interface**:

   - Voice recording button
   - Example phrases
   - Results display
3. **Results Display**:

   - Recognized speech
   - Detected intent
   - Banking operation response
   - Account balances or transaction history

```javascript
// Voice recording functionality
function startRecording() {
    // Reset previous results
    resultSection.classList.add('hidden');
  
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            isRecording = true;
            recordBtn.classList.add('recording');
            recordText.textContent = 'Stop Recording';
            recordingIndicator.classList.remove('hidden');
      
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
      
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
      
            mediaRecorder.onstop = processRecording;
            mediaRecorder.start();
        });
}
```

---

## Multilingual Support

The system supports multiple languages with these components:

1. **Language Configuration**:

   - User language preferences stored in user profiles
   - Language-specific configuration in `azure_config.json`
2. **Speech Recognition**:

   - Language-specific service selection
   - Azure Speech Services for Hindi and Swahili
   - Google Speech Recognition for English
3. **Intent Recognition**:

   - Language-specific patterns in `intent_patterns.json`
   - Separate extractors for each language
   - Keyword fallbacks in different languages
4. **User Interface**:

   - Language-specific example phrases
   - Language selection dropdown

---

## Error Handling and Resilience

The system implements comprehensive error handling:

1. **Speech Recognition**:

   - Multiple fallback systems if primary recognition fails
   - Service-specific error handling
   - Clear error messages for troubleshooting
2. **Audio Processing**:

   - Multiple audio conversion methods
   - Format validation and error handling
   - Cleanup of temporary files
3. **Frontend Error Handling**:

   - User-friendly error messages
   - Visual feedback during processing
   - Connection error handling
4. **Security Measures**:

   - Voice authentication with confidence scores
   - Secure password storage with hashing
   - Input validation

---

## Data Flow Diagram

```
┌────────────┐        ┌──────────────┐        ┌───────────────────┐
│            │        │              │        │                   │
│   Client   ├────────► Flask Server ├────────► Speech Recognition │
│            │        │              │        │                   │
└─────┬──────┘        └─────┬────────┘        └─────────┬─────────┘
      │                     │                           │
      │                     │                           │
      │                     ▼                           ▼
┌─────┴──────┐        ┌──────────────┐        ┌───────────────────┐
│            │        │              │        │                   │
│  Response  │◄───────┤ Banking Ops  │◄───────┤  Intent Detection  │
│            │        │              │        │                   │
└────────────┘        └──────────────┘        └───────────────────┘
```

---

## Authentication Workflow

1. **User Registration**:

   - User submits registration details
   - System creates user profile
   - Password is hashed and stored
2. **Login Process**:

   - User provides credentials
   - System verifies password hash
   - User details stored in local storage
   - UI updates to show banking interface
3. **Voice Authentication**:

   - User provides voice sample
   - System compares with stored voice print
   - Authentication result determines access

```
┌──────────┐      ┌───────────┐      ┌──────────────┐      ┌────────────┐
│          │      │           │      │              │      │            │
│   User   ├─────►│  Login    ├─────►│ Authenticate ├─────►│ Banking UI │
│          │      │           │      │              │      │            │
└──────────┘      └───────────┘      └──────────────┘      └────────────┘
```

---

## Voice Processing Workflow

1. **Record Audio**:

   - Frontend captures audio using MediaRecorder API
   - Audio blob sent to server
2. **Processing Pipeline**:

   - Audio converted to appropriate format
   - Voice authentication performed
   - Speech recognition converts to text
   - Intent extraction identifies operation
   - Banking service executes operation
3. **Response Handling**:

   - Banking operation results returned to client
   - UI updated to display results

```
┌──────────┐      ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│          │      │             │      │              │      │             │
│  Record  ├─────►│  Recognize  ├─────►│  Extract     ├─────►│   Process   │
│  Audio   │      │  Speech     │      │  Intent      │      │   Banking   │
│          │      │             │      │              │      │   Request   │
└──────────┘      └─────────────┘      └──────────────┘      └─────────────┘
```

---

## Banking Operation Workflow

1. **Balance Check**:

   - Voice command processed
   - User account identified
   - Balances retrieved from mock database
   - Results formatted and returned
2. **Money Transfer**:

   - Amount and recipient extracted from voice command
   - Sender and recipient accounts identified
   - Balance verification
   - Transaction records created for both parties
   - Balances updated
3. **Transaction History**:

   - Time period extracted (recent, last week, last month)
   - Transactions filtered by date
   - Limited set returned to client

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│             │      │              │      │                │
│  Intent     ├─────►│  Retrieve    ├─────►│  Update        │
│  Detection  │      │  User Data   │      │  Database      │
│             │      │              │      │                │
└─────────────┘      └──────────────┘      └────────────────┘
        │                                          │
        │                                          │
        │                                          ▼
┌───────┴───────┐                          ┌────────────────┐
│               │                          │                │
│  Format       │◄─────────────────────────┤  Return        │
│  Response     │                          │  Response      │
│               │                          │                │
└───────────────┘                          └────────────────┘
```

---

## Technical Implementation Details

### Speech Recognition Configuration

The tiered approach to speech recognition is configured in `azure_config.json`:

```json
{
  "language_mapping": {
    "en-US": "en-US",
    "hi-IN": "hi-IN", 
    "sw": "sw-KE",
    "default": "en-US"
  },
  "fallback_priority": ["azure", "google", "local"]
}
```

This configuration:

- Maps user language preferences to service-specific language codes
- Defines the order of fallback services if the primary service fails

---

### Intent Recognition Patterns

The `intent_patterns.json` file defines language-specific regex patterns:

```json
{
  "en-US": {
    "check_balance": {
      "patterns": [
        "what(('s)|( is)) my balance",
        "check (my )?(bank |account )?balance",
        "how much (money )?(do i have|is in my account)",
        // Additional patterns
      ]
    },
    // Additional intents
  },
  "hi-IN": {
    "check_balance": {
      "patterns": [
        "मेरा बैलेंस क्या है",
        "बैलेंस चेक करें",
        // Additional patterns
      ]
    },
    // Additional intents
  }
}
```

These patterns are compiled into regex expressions for efficient matching.

---

### Audio Processing and Conversion

The system handles various audio formats through a robust conversion pipeline:

1. **Format Detection**:

   - Identifies input audio format
   - Validates file integrity
2. **Conversion Methods**:

   - Primary: pydub with format detection
   - Secondary: pydub with auto-detection
   - Tertiary: ffmpeg direct call
   - Quaternary: librosa-based conversion
3. **Output Format**:

   - 16kHz WAV format for best compatibility with all services
   - Single channel (mono) for speech recognition

---

### Voice Biometrics Technical Details

The voice authentication system uses these techniques:

1. **MFCC Feature Extraction**:

   - 13 Mel-frequency cepstral coefficients
   - Normalization for consistency
   - Feature matrix transposed for GMM model compatibility
2. **Gaussian Mixture Models**:

   - 16 mixture components
   - Diagonal covariance matrix
   - Maximum 200 iterations for convergence
3. **Authentication Decision**:

   - Log-likelihood score compared to threshold
   - Default threshold (-80) set for demonstration purposes
   - Real-world systems would use more sophisticated thresholding

---

## Frontend Implementation

The frontend is built with:

1. **Core Technologies**:

   - HTML5 for structure
   - CSS3 for styling
   - JavaScript for interactivity
2. **Key Features**:

   - Responsive design for different devices
   - Dynamic content updating
   - Audio recording via MediaRecorder API
   - AJAX communication with the backend
3. **State Management**:

   - User session maintained in localStorage
   - Voice recording state tracked in memory
   - Dynamic UI updates based on application state

---

## Complete System Workflow

```
┌─────────────────┐      ┌───────────────┐      ┌─────────────────┐
│ User Interface  │      │ Flask Server  │      │ Speech & Intent │
├─────────────────┤      ├───────────────┤      ├─────────────────┤
│ 1. Login        │─────►│ Authentication│─────►│                 │
│ 2. Record Audio │─────►│ Process Audio │─────►│ Recognize Speech│─┐
│                 │      │               │      │                 │ │
│                 │      │               │      │ Extract Intent  │◄┘
│                 │      │               │      └────────┬────────┘
│                 │      │               │               │
│                 │      │               │               ▼
│                 │      │               │      ┌─────────────────┐
│                 │      │               │      │ Banking Service │
│                 │      │               │      ├─────────────────┤
│                 │      │               │      │ Process Request │◄─┐
│                 │      │               │      │                 │  │
│                 │      │               │      │ Update Database │──┘
│                 │      │               │      └────────┬────────┘
│                 │      │               │               │
│                 │◄─────│ Format Result │◄──────────────┘
│ 3. Display      │      │               │
│    Results      │      │               │
└─────────────────┘      └───────────────┘
```

---

## Deployment Options

The system can be deployed in several ways:

1. **Local Development**:

   - Run with `python app.py`
   - Access at `http://127.0.0.1:5000`
2. **Cloud Deployment**:

   - Deploy to cloud platforms like Azure, AWS, or Google Cloud
   - Set up proper environment variables for API keys
3. **Containerization**:

   - Package with Docker for consistent deployment
   - Define services in Docker Compose for orchestration

---

## Future Enhancements

Potential improvements to the system:

1. **Speech Recognition**:

   - Fine-tune custom models for specific languages
   - Implement noise cancellation and speech enhancement
2. **Voice Biometrics**:

   - Implement anti-spoofing measures
   - Use deep learning models for better authentication
3. **Banking Features**:

   - Add more banking operations
   - Implement real banking API integrations
4. **Security**:

   - Implement proper session management
   - Add two-factor authentication
5. **Offline Support**:

   - Enhance local models for better offline performance
   - Add offline mode for low-connectivity areas

---
