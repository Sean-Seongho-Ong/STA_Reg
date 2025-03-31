import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  CssBaseline, 
  Paper,
  AppBar,
  Toolbar,
  ThemeProvider,
  createTheme
} from '@mui/material';
import ChatInterface from './components/ChatInterface';
import './App.css';

// 테마 설정
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static" color="primary">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              QLoRA LLM 챗봇
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Container maxWidth="md" sx={{ mt: 4, mb: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 2, 
                flexGrow: 1, 
                display: 'flex',
                flexDirection: 'column',
                minHeight: 'calc(100vh - 180px)'
              }}
            >
              <ChatInterface />
            </Paper>
          </Box>
        </Container>
        
        <Box component="footer" sx={{ p: 2, bgcolor: 'background.paper', mt: 'auto' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            © {new Date().getFullYear()} QLoRA LLM 챗봇 | Powered by Llama-2-13B
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
