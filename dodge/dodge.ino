#include <Car_Library.h>

int analogPin = A6;
int motorA1 = 2;    // 모터 드라이버 IN1
int motorA2 = 3;    // 모터 드라이버 IN2
int motorB1 = 4;
int motorB2 = 5;
int motorH1 = 6;
int motorH2 = 7;
int trig = 8;
int echo = 9;
int reg;
int temp;
long distance;
int count = 0;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);       // 시리얼 통신 시작, 통신 속도 설정
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);
  pinMode(analogPin, INPUT);
  pinMode(trig, OUTPUT);    // trig 핀 모드 설정
  pinMode(echo, INPUT);     // echo 핀 모드 설정

  reg = potentiometer_Read(analogPin);
  if(reg < 17) {
    while(reg < 17) {
      motor_forward(motorH1, motorH2, 150);
      delay(10);
      motor_hold(motorH1, motorH2);
      reg = potentiometer_Read(analogPin);
    }
  }
  else if(reg > 17) {
    while(reg > 17) {
      motor_backward(motorH1, motorH2, 150);
      delay(10);
      motor_hold(motorH1, motorH2);
      reg = potentiometer_Read(analogPin);
    }
  }
}

void loop() {
  distance = ultrasonic_distance(trig, echo);
  Serial.print(distance);
  Serial.println(" mm");
  delay(100);

  if(distance < 1000 && count == 0) {
      motor_forward(motorH1, motorH2, 150);
      delay(3000);
      motor_hold(motorH1, motorH2);
      motor_backward(motorH1, motorH2, 150);
      delay(3000);
      motor_hold(motorH1, motorH2);
      reg = potentiometer_Read(analogPin);
      while(reg < 17) {
          motor_forward(motorH1, motorH2, 150);
          delay(10);
          motor_hold(motorH1, motorH2);
          reg = potentiometer_Read(analogPin);
        }
      count++;
    }
    else if(distance < 1000 && count == 1) {
      motor_backward(motorH1, motorH2, 150);
      delay(3000);
      motor_hold(motorH1, motorH2);
      motor_forward(motorH1, motorH2, 150);
      delay(3000);
      motor_hold(motorH1, motorH2);
      reg = potentiometer_Read(analogPin);
      while(reg > 17) {
          motor_backward(motorH1, motorH2, 150);
          delay(10);
          motor_hold(motorH1, motorH2);
          reg = potentiometer_Read(analogPin);
        }
      count--;
    }
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) { // 시리얼 버퍼에 데이터가 있는지 확인
    temp = Serial.parseInt(); // 정수형 데이터 읽기
    Serial.print("Received Value: ");
    Serial.println(temp); // 받은 값 출력
    
    if(temp == 999) {
      motor_hold(motorA1, motorA2);
      motor_hold(motorB1, motorB2);
    }
    else if(temp == 300) {
      motor_forward(motorA1, motorA2, 150);
      motor_forward(motorB1, motorB2, 150);
    }
    else {
      motor_forward(motorA1, motorA2, 150);
      motor_forward(motorB1, motorB2, 150);

        reg = potentiometer_Read(analogPin);
      if (temp < 10)
        temp = 10;
      if (temp > 30)
        temp = 30;
      if (temp > reg) {
        while(temp > reg) {
          motor_forward(motorH1, motorH2, 150);
          delay(10);
          motor_hold(motorH1, motorH2);
          reg = potentiometer_Read(analogPin);
        }
      }
      else {
        while(temp < reg) {
          motor_backward(motorH1, motorH2, 150);
          delay(10);
          motor_hold(motorH1, motorH2);
          reg = potentiometer_Read(analogPin);
        }
      }
      motor_hold(motorH1, motorH2);
    }
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}
