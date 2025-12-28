import { useState, useEffect } from 'react';

export const useDeviceType = () => {
  const [isIOS, setIsIOS] = useState(false);
  const [isAndroid, setIsAndroid] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const ua = navigator.userAgent.toLowerCase();

    // iPhone, iPad, iPod かどうか
    const isIosDevice = /iphone|ipad|ipod/.test(ua);
    
    // Android かどうか
    const isAndroidDevice = /android/.test(ua);

    setIsIOS(isIosDevice);
    setIsAndroid(isAndroidDevice);
    setIsMobile(isIosDevice || isAndroidDevice);

  }, []);

  return { isIOS, isAndroid, isMobile };
};