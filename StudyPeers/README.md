# 🎓 StudyPeers - CSE 3310 Team Project

**Mobile Application for University Peer Study Matching & Collaboration**

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Project Name** | StudyPeers |
| **Course** | CSE 3310: Software Requirement Analysis & Testing |
| **Semester** | Spring 2026 |
| **Team Number** | 15 |
| **Deliverable** | Increment II - Complete Test Plan (v1.0) |
| **Submission Date** | April 16, 2026 |

---

## 🎯 Project Vision

**StudyPeers** is a mobile application that revolutionizes how university students find study partners and collaborate academically. The platform uses intelligent matching algorithms, real-time communication, and integrated AI tutoring to create a comprehensive learning ecosystem.

### Key Features

✅ **Intelligent Peer Matching** - Find compatible study partners based on courses, availability, learning styles  
✅ **Session Scheduling** - Coordinate and manage study sessions with detailed requests and confirmations  
✅ **Real-time Chat** - Communicate with matched study partners and exchange resources  
✅ **AI Tutor Integration** - Access Claude AI for instant academic support and homework help  
✅ **Rating & Feedback** - Build reputation through peer ratings and session reviews  
✅ **Session History** - Track past sessions and build long-term study relationships  
✅ **Push Notifications** - Stay updated with real-time alerts for matches, messages, and sessions  

---

## 👥 Team Members & Roles

| Member | Role | Expertise |
|--------|------|-----------|
| **Mohammed Usman Khan** | Lead Developer | Full-Stack Development, Backend Architecture |
| **Aayush Thapaliya** | Frontend Lead | React Native, UI/UX Design |
| **Bhawana Chapagain** | QA Lead & Documentation | Testing, Documentation, Data Analysis |
| **Steven Phi Hung Dang** | Backend Developer | Node.js/Express, Database Design |

---

## 🏗️ Technology Stack

### Frontend
- **Framework**: React Native
- **Language**: JavaScript/TypeScript
- **UI Library**: React Navigation, Native Base

### Backend
- **Runtime**: Node.js
- **Framework**: Express.js
- **Language**: JavaScript

### Database
- **Primary**: SQLite (Local Development)
- **ORM**: Sequelize

### External Services
- **AI Tutoring**: Anthropic Claude API
- **Push Notifications**: Firebase Cloud Messaging (FCM)
- **Email Service**: SMTP (Gmail/SendGrid)

### Development Tools
- **Version Control**: Git/GitHub
- **Testing**: Jest, React Native Testing Library
- **Code Editor**: VS Code

---

## 📊 Component Overview (13 Functional Components)

| # | Component | Test Cases | Status |
|---|-----------|-----------|--------|
| 1 | Member Registration | 7 | ✅ Complete |
| 2 | Login | 7 | ✅ Complete |
| 3 | Profile Setup & Editing | 7 | ✅ Complete |
| 4 | Courses & University Information | 6 | ✅ Complete |
| 5 | Match Generation & Swiping | 7 | ✅ Complete |
| 6 | Study Session Request | 6 | ✅ Complete |
| 7 | Session Management | 7 | ✅ Complete |
| 8 | Session Feedback & Ratings | 7 | ✅ Complete |
| 9 | Session History | 6 | ✅ Complete |
| 10 | Match Search & Filters | 6 | ✅ Complete |
| 11 | Chat Messaging | 7 | ✅ Complete |
| 12 | AI Tutor | 7 | ✅ Complete |
| 13 | Notification System | 7 | ✅ Complete |
| **TOTAL** | **85+ Test Cases** | | **✅ Complete** |

---

## 📁 Repository Structure

```
StudyPeers/
├── README.md                          # This file
├── TEAM_MEMBERS.md                    # Detailed team member info
├── docs/
│   ├── TEST_PLAN.md                  # Complete Test Plan (85+ test cases)
│   ├── REQUIREMENTS.md               # Functional requirements
│   ├── ARCHITECTURE.md               # System architecture diagram
│   └── API_DOCUMENTATION.md          # API endpoints
├── src/
│   ├── frontend/                     # React Native app
│   ├── backend/                      # Node.js/Express server
│   └── database/                     # SQLite schema
├── test/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── e2e/                          # End-to-end tests
├── config/
│   ├── env.example                   # Environment variables template
│   └── database.config.js            # Database configuration
├── .gitignore                         # Git ignore file
└── package.json                      # Dependencies

```

---

## 🚀 Getting Started

### Prerequisites
- Node.js v16.x or higher
- npm or yarn package manager
- Android/iOS development environment (for mobile testing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhawanachapagain/StudentMLProject.git
   cd StudentMLProject/StudyPeers
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   cp config/env.example .env
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

---

## 📚 Documentation

### Key Documents
- **[TEST_PLAN.md](./docs/TEST_PLAN.md)** - Complete test plan with all 85+ test cases
- **[TEAM_MEMBERS.md](./TEAM_MEMBERS.md)** - Team roles and responsibilities
- **API Documentation** - RESTful API endpoint specifications
- **Architecture Diagram** - System design and data flow

---

## ✅ Testing Strategy

### Test Coverage
- **Unit Tests**: Individual component and function testing
- **Integration Tests**: Component interaction and API testing
- **End-to-End Tests**: Full user journey testing on mobile devices
- **Functional Tests**: Verification against requirements

### Test Environment
- Local development server (localhost)
- Test database with seed data
- Physical Android/iOS devices for FCM testing
- Pre-created test user accounts

---

## 🔄 Development Workflow

1. **Create feature branch** from `studypeers-team15`
   ```bash
   git checkout -b feature/component-name
   ```

2. **Implement feature** with unit tests

3. **Run tests locally**
   ```bash
   npm test
   ```

4. **Create Pull Request** to `studypeers-team15`

5. **Code Review** by team members

6. **Merge** to main branch

---

## 📈 Project Timeline

| Phase | Dates | Deliverable |
|-------|-------|-------------|
| **Requirements Analysis** | Week 1-2 | SRA Document |
| **Design & Planning** | Week 3-4 | Architecture Design |
| **Development (Increment II)** | Week 5-8 | Functional Build |
| **Testing** | Week 9-10 | Test Plan & Results |
| **Final Submission** | Week 11 | Complete Package |

---

## 🔐 Key Features Details

### 1. Member Registration
- .edu email verification
- Password strength validation
- University and major selection
- Account creation with email confirmation

### 2. Login & Authentication
- JWT token-based authentication
- Account lockout after 3 failed attempts
- Email verification requirement
- Session management

### 3. Profile Management
- Complete student profile (bio, learning style, availability)
- Profile picture upload
- Course management (current & past)
- Preference settings

### 4. Intelligent Matching
- Compatibility scoring algorithm
- Shared course detection
- Availability overlap calculation
- Like/Pass swiping interface
- Block user functionality

### 5. Study Session Coordination
- Session request with proposed times
- Accept/Decline/Counter-offer workflow
- Session mode specification (Online/In-Person)
- Location tracking for in-person sessions

### 6. Real-time Communication
- One-to-one chat messaging
- Image sharing in conversations
- Message persistence
- Unread message indicators

### 7. AI Tutor
- Integration with Claude API
- Text and image-based queries
- Conversation history
- Loading indicators during API calls

### 8. Ratings & Feedback
- 1-5 star rating system
- Optional comment field
- Average rating calculation
- Duplicate submission prevention

### 9. Notifications
- Firebase Cloud Messaging (FCM)
- Multiple notification types (NEW_MATCH, SESSION_REQUEST, MESSAGE_RECEIVED, etc.)
- In-app notification list
- Push notification navigation

---

## 🎯 Success Criteria

✅ All 13 components fully functional  
✅ 85+ test cases defined and passing  
✅ User authentication secure and working  
✅ Real-time features operational  
✅ AI Tutor integration successful  
✅ Push notifications delivering correctly  
✅ Database persisting data reliably  
✅ Mobile app responsive on Android/iOS  

---

## 🤝 Contributing Guidelines

1. Follow Git workflow (feature branches)
2. Write unit tests for new features
3. Document your code with comments
4. Update README if adding new features
5. Get code review before merging
6. Maintain consistent code style

---

## 📞 Support & Communication

- **Primary Channel**: Team meetings (Mondays & Wednesdays)
- **Discussion**: GitHub Issues and Pull Requests
- **Documentation**: This README and docs/ folder
- **Emergency Contact**: Team lead (Mohammed Usman Khan)

---

## 📝 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 4/16/2026 | Initial Test Plan document with all 13 component test tables |

---

## 📄 License

This project is part of CSE 3310 coursework at University of Texas at Arlington.

---

## ⭐ Acknowledgments

Special thanks to our QA Lead **Bhawana Chapagain** for comprehensive testing documentation and our entire team for dedication to this project.

**Last Updated:** May 13, 2026  
**Status**: ✅ Increment II Complete

