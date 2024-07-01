#include <Car_Library.h>

int analogPin = A6;
int motorA1 = 2;    // 모터 드라이버 IN1
int motorA2 = 3;    // 모터 드라이버 IN2
int motorB1 = 4;
int motorB2 = 5;
int motorH1 = 6;
int motorH2 = 7;
int reg;
int temp;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);       // 시리얼 통신 시작, 통신 속도 설정
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);
  pinMode(analogPin, INPUT);
}

void loop() {

  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) { // 시리얼 버퍼에 데이터가 있는지 확인
    temp = Serial.parseInt(); // 정수형 데이터 읽기
    Serial.flush();
    Serial.print("Received Value: ");
    Serial.println(temp); // 받은 값 출력
    
    if(temp == 0) {
      motor_hold(motorA1, motorA2);
      motor_hold(motorB1, motorB2);
    }
    else {
      motor_forward(motorA1, motorA2, 100);
      motor_forward(motorB1, motorB2, 100);

      reg = potentiometer_Read(analogPin);
      Serial.print("Value : ");
      Serial.println(reg);
      
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

    while (Serial.available() > 0)
      Serial.read();
  }


/*
  // Backward
  Serial.println("Motor Backward");
  motor_backward(motorA1, motorA2, 75);
  motor_backward(motorB1, motorB2, 75);
  delay(3000);

  // Hold
  Serial.println("Motor Hold");
  motor_hold(motorA1, motorA2);
  motor_hold(motorB1, motorB2);
  delay(3000);
*/
}
