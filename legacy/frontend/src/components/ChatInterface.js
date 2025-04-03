import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  IconButton,
  CircularProgress,
  Avatar,
  Alert,
  Snackbar
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import config from '../config';

// 메시지 컴포넌트
const Message = React.memo(({ message, isBot }) => {
  // 메시지 전처리: HTML 태그 제거 및 텍스트 정리
  const processedMessage = React.useMemo(() => {
    if (!message) return '';
    
    // 디버깅 정보 출력
    console.log("처리할 메시지:", JSON.stringify(message).substring(0, 150));
    
    let cleanedMessage = message;
    let debugInfo = { original: message.substring(0, 50) };
    
    try {
      // 1. HTML 태그 처리
      if (cleanedMessage.includes("<RESPONSE_START>") && cleanedMessage.includes("</RESPONSE_END>")) {
        const start = cleanedMessage.indexOf("<RESPONSE_START>") + "<RESPONSE_START>".length;
        const end = cleanedMessage.indexOf("</RESPONSE_END>");
        if (start !== -1 && end !== -1 && start < end) {
          console.log(`HTML 태그 제거: 시작=${start}, 끝=${end}`);
          
          // 태그 제거 전 원본 텍스트 저장
          debugInfo.beforeTagRemoval = cleanedMessage.substring(start, Math.min(start + 50, end));
          
          cleanedMessage = cleanedMessage.substring(start, end);
          debugInfo.afterTagRemoval = cleanedMessage.substring(0, 50);
        }
      }
      
      // 2. 남아있는 태그 관련 텍스트 제거
      const tagPatterns = [
        "E_START>", "<RESPONS", "SPONSE_", "_START", "START>", "ESPONSE", 
        "<RESPONSE_START>", "</RESPONSE_END>", "ESPONSE_END", "ESPONSE>", "/RESPONSE", "PONSE>"
      ];
      
      tagPatterns.forEach(pattern => {
        if (cleanedMessage.includes(pattern)) {
          console.log(`부분 태그 '${pattern}' 제거`);
          debugInfo[`before_${pattern}_removal`] = cleanedMessage.substring(0, 50);
          cleanedMessage = cleanedMessage.replace(new RegExp(pattern, 'g'), '');
          debugInfo[`after_${pattern}_removal`] = cleanedMessage.substring(0, 50);
        }
      });
      
      // 3. 특수 문자 접두사 제거
      if (cleanedMessage.startsWith("■★▶︎ ")) {
        console.log("특수 문자 접두사 제거");
        debugInfo.beforePrefixRemoval = cleanedMessage.substring(0, 50);
        cleanedMessage = cleanedMessage.substring(5);
        debugInfo.afterPrefixRemoval = cleanedMessage.substring(0, 50);
      }
      
      // 4. 이전 형식 접두사 제거
      if (cleanedMessage.startsWith("#응답내용#")) {
        console.log("기존 응답내용 접두사 제거");
        debugInfo.beforeOldPrefixRemoval = cleanedMessage.substring(0, 50);
        cleanedMessage = cleanedMessage.substring(10);
        debugInfo.afterOldPrefixRemoval = cleanedMessage.substring(0, 50);
      }
      
      console.log("정리된 메시지:", cleanedMessage.substring(0, 50));
      console.log("디버깅 정보:", debugInfo);
      return cleanedMessage;
    } catch (error) {
      console.error("메시지 처리 중 오류 발생:", error);
      return message; // 오류 발생 시 원본 메시지 반환
    }
  }, [message]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'row',
        mb: 2,
        alignItems: 'flex-start',
      }}
    >
      <Avatar
        sx={{
          bgcolor: isBot ? 'primary.main' : 'secondary.main',
          mr: 1,
        }}
      >
        {isBot ? <SmartToyIcon /> : <PersonIcon />}
      </Avatar>
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: '85%',
          bgcolor: isBot ? 'grey.100' : 'primary.light',
          color: isBot ? 'text.primary' : 'primary.contrastText',
          borderRadius: 2,
          borderTopLeftRadius: isBot ? 0 : 2,
          borderTopRightRadius: isBot ? 2 : 0,
        }}
      >
        {isBot ? (
          <>
            {/* Typography 컴포넌트 사용 */}
            <Typography 
              variant="body1" 
              component="div"
              sx={{ 
                whiteSpace: 'pre-wrap',
                '& p': { mt: 1, mb: 1 },
                '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1 },
                '& a': { color: 'primary.main' },
                '& code': { 
                  backgroundColor: 'rgba(0, 0, 0, 0.05)', 
                  padding: '2px 4px', 
                  borderRadius: '4px',
                  fontFamily: 'monospace'
                },
                '& pre': {
                  backgroundColor: 'rgba(0, 0, 0, 0.05)',
                  padding: '8px',
                  borderRadius: '4px',
                  overflowX: 'auto',
                  '& code': {
                    padding: 0,
                    backgroundColor: 'transparent'
                  }
                },
                '& ul, & ol': { pl: 3 }
              }}
            >
              {processedMessage}
            </Typography>
            
            {/* 디버깅용 정보 - 개발 완료 후 제거 */}
            {process.env.NODE_ENV === 'development' && (
              <Box mt={1} p={1} bgcolor="rgba(0,0,0,0.05)" borderRadius={1}>
                <Typography variant="caption" color="textSecondary">
                  메시지 길이: {processedMessage?.length || 0}자
                  {processedMessage?.length > 0 && `, 처음 50자: "${processedMessage.substring(0, 50).replace(/\n/g, '\\n')}"`}
                </Typography>
              </Box>
            )}
          </>
        ) : (
          <Typography variant="body1">{processedMessage}</Typography>
        )}
      </Paper>
    </Box>
  );
});

Message.displayName = 'Message';

// 로딩 메시지 컴포넌트
const LoadingMessage = React.memo(() => (
  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
    <Avatar sx={{ bgcolor: 'primary.main', mr: 1 }}>
      <SmartToyIcon />
    </Avatar>
    <Paper
      elevation={1}
      sx={{
        p: 2,
        maxWidth: '85%',
        bgcolor: 'grey.100',
        borderRadius: 2,
        borderTopLeftRadius: 0,
        display: 'flex',
        alignItems: 'center'
      }}
    >
      <CircularProgress size={20} sx={{ mr: 1 }} />
      <Typography variant="body2" color="text.secondary">
        응답 생성 중...
      </Typography>
    </Paper>
  </Box>
));

LoadingMessage.displayName = 'LoadingMessage';

// 챗 인터페이스 컴포넌트
const ChatInterface = () => {
  // 상태 관리
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('connecting');
  const [errorMessage, setErrorMessage] = useState('');
  const [showError, setShowError] = useState(false);
  const messagesEndRef = useRef(null);

  // 채팅창 자동 스크롤
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // API 상태 확인
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        // 백엔드 API 상태 확인
        const response = await axios.get(`${config.apiBaseUrl}/api/test`);
        if (response.status === 200) {
          setApiStatus('connected');
        }
      } catch (error) {
        console.error('API 연결 오류:', error);
        setApiStatus('error');
        setErrorMessage('백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.');
        setShowError(true);
      }
    };

    checkApiStatus();
  }, []);

  // 채팅 메시지 전송
  const sendMessage = useCallback(async () => {
    if (inputText.trim() === '') return;

    const userMessage = inputText.trim();
    setInputText('');
    
    // 사용자 메시지 추가
    setMessages((prevMessages) => [
      ...prevMessages,
      { role: 'user', content: userMessage },
    ]);
    
    setIsLoading(true);

    try {
      // API 호출
      const response = await axios.post(`${config.apiBaseUrl}/api/chat`, {
        messages: [
          ...messages.map(msg => ({ role: msg.role, content: msg.content })),
          { role: 'user', content: userMessage }
        ],
        temperature: 0.7,
        max_tokens: 2048
      }, {
        timeout: 120000 // 120초 타임아웃 설정 (응답 시간이 길어질 수 있음)
      });

      // 응답 데이터 안전하게 처리
      let responseText = '';
      if (response && response.data && response.data.response) {
        responseText = response.data.response;
        
        // 디버깅을 위해 원본 응답 로깅
        console.log("원본 응답:", JSON.stringify(responseText).substring(0, 150));
        console.log("원본 응답 길이:", responseText.length);
        
        // 응답 형식이 없거나 비어있는 경우 처리
        if (!responseText || responseText.trim() === '') {
          responseText = "응답이 비어있습니다.";
          console.error("빈 응답 감지됨");
        } else {
          // 태그가 중첩된 응답 감지 및 처리
          if (responseText.includes("E_START>") || 
              (responseText.includes("<RESPONSE_START>") && responseText.indexOf("<RESPONSE_START>") !== 0)) {
            console.warn("태그 중첩 감지됨, 특수 처리 적용");
            
            // 중첩된 태그 제거
            const cleanedText = responseText
              .replace(/<RESPONSE_START>/g, "")
              .replace(/<\/RESPONSE_END>/g, "")
              .replace(/E_START>/g, "")
              .replace(/_START>/g, "")
              .replace(/<RESPONS/g, "")
              .replace(/ESPONSE_END/g, "")
              .replace(/ESPONSE>/g, "")
              .replace(/\/RESPONSE/g, "")
              .replace(/PONSE>/g, "");
            
            // 다시 올바른 태그로 감싸기
            responseText = `<RESPONSE_START>${cleanedText}</RESPONSE_END>`;
            console.log("정리된 응답:", responseText.substring(0, 150));
          }
        }
      } else {
        console.error("응답 형식 오류:", response);
        responseText = "응답 형식이 올바르지 않습니다.";
      }

      // 봇 응답 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'assistant', content: responseText },
      ]);
    } catch (error) {
      console.error('Error fetching response:', error);
      let errorMsg = '죄송합니다. 응답을 처리하는 동안 오류가 발생했습니다.';
      
      if (error.response) {
        // 서버에서 응답이 왔지만 오류가 있는 경우
        errorMsg += ` (${error.response.status}: ${error.response.data.detail || '알 수 없는 오류'})`;
      } else if (error.request) {
        // 요청은 보냈지만 응답이 없는 경우
        errorMsg += ' 서버에서 응답이 없습니다. 백엔드 서버가 실행 중인지 확인하세요.';
      }
      
      // 에러 메시지 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'assistant', content: errorMsg },
      ]);
      
      setErrorMessage(errorMsg);
      setShowError(true);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, messages]);

  // Enter 키 처리
  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  // 채팅 초기화
  const clearChat = useCallback(() => {
    setMessages([]);
  }, []);

  // 에러 알림 닫기
  const handleCloseError = useCallback(() => {
    setShowError(false);
  }, []);

  // 입력 텍스트 변경 처리
  const handleInputChange = useCallback((e) => {
    setInputText(e.target.value);
  }, []);

  // 연결 상태 표시
  const renderConnectionStatus = () => {
    if (apiStatus === 'connected') {
      return (
        <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>
          ● 서버에 연결되었습니다
        </Typography>
      );
    } else if (apiStatus === 'error') {
      return (
        <Typography variant="body2" color="error.main" sx={{ mt: 1 }}>
          ● 서버 연결 오류
        </Typography>
      );
    } else {
      return (
        <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
          ● 서버 연결 확인 중...
        </Typography>
      );
    }
  };

  // 빈 채팅 화면 표시
  const renderEmptyChat = () => (
    <Box 
      sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        height: '100%',
        flexDirection: 'column'
      }}
    >
      <SmartToyIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
      <Typography variant="body1" color="text.secondary" align="center">
        무엇이든 질문해보세요!
      </Typography>
      {apiStatus === 'error' && (
        <Typography variant="body2" color="error.main" sx={{ mt: 2 }}>
          백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.
        </Typography>
      )}
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* 에러 알림 */}
      <Snackbar 
        open={showError} 
        autoHideDuration={6000} 
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {errorMessage}
        </Alert>
      </Snackbar>

      {/* 채팅 헤더 */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h6" component="div">
          QLoRA LLM 챗봇과 대화하기
        </Typography>
        <Typography variant="body2" color="text.secondary">
          질문을 입력하시면 학습된 LLM 모델이 답변을 제공합니다.
        </Typography>
        {renderConnectionStatus()}
      </Box>
      
      {/* 메시지 영역 */}
      <Box sx={{ 
        flexGrow: 1, 
        overflow: 'auto', 
        p: 2,
        display: 'flex',
        flexDirection: 'column'
      }}>
        {messages.length === 0 
          ? renderEmptyChat()
          : messages.map((msg, index) => (
              <Message
                key={index}
                message={msg.content}
                isBot={msg.role === 'assistant'}
              />
            ))
        }
        
        {isLoading && <LoadingMessage />}
        
        <div ref={messagesEndRef} />
      </Box>
      
      {/* 입력 영역 */}
      <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <TextField
            fullWidth
            placeholder="메시지를 입력하세요..."
            variant="outlined"
            value={inputText}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            multiline
            maxRows={4}
            sx={{ mr: 1 }}
            disabled={isLoading || apiStatus === 'error'}
          />
          <Button
            variant="contained"
            color="primary"
            endIcon={<SendIcon />}
            onClick={sendMessage}
            disabled={inputText.trim() === '' || isLoading || apiStatus === 'error'}
          >
            전송
          </Button>
          <IconButton 
            color="error" 
            onClick={clearChat} 
            sx={{ ml: 1 }}
            disabled={messages.length === 0 || isLoading}
          >
            <DeleteIcon />
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatInterface; 